import re
import secrets as pysecrets
import shlex
import time

from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync

import runpod

RUNPOD_TEMPLATE_ID = "1a33vbssq9"
# Container image for the pod. The template above carries ports/env but no
# image (RunPod rejects create_pod with an empty image_name), so it must be set
# explicitly. Override here if you build/use a different OneTrainer image.
RUNPOD_IMAGE = "dxqbyd/runpod-onetrainer-cli:1.2"
# RunPod selects by *data center id*, not country code. The old country-code
# list ("EU", "PL", …) never matched a real DC, so create_pod always reported
# "no instances available" even when the GPU was in stock.
# Restricted to European data centers certified ISO/IEC 27001 or SOC 2 Type II, since the uploaded latent cache
# decodes back to the training images. From RunPod's data center catalog (compliance field), 2026-10-12:
#   EU-NL-1, EUR-IS-1: GDPR, ISO 27001 | EUR-IS-3/4/5: GDPR, SOC 2 Type II | EU-SE-1: ISO 27001
# Left out on purpose: EU-RO-1, EU-CZ-1, EUR-IS-2, EUR-NO-1 (GDPR only) and EU-FR-1, EUR-NO-2 (none listed).
# The certification covers the data center itself, so it only means something for SECURE cloud pods.
RUNPOD_DATA_CENTERS = ["EU-NL-1", "EUR-IS-1", "EUR-IS-3", "EUR-IS-4", "EUR-IS-5", "EU-SE-1"]
# Ephemeral container disk (wiped on stop). All real storage — cache, outputs,
# backups, HF cache — lives on the persistent volume mounted at /workspace, so
# this only needs room for the OS/scratch.
RUNPOD_CONTAINER_DISK_GB = 30
# Host CUDA versions the pod may land on. The cu130 torch build needs a host driver that reports CUDA 13.x;
# without this RunPod can place the pod on an older host, where torch fails at startup. The v1 API behind
# runpod.create_pod only takes an exact allow-list (no minimum), so list every 13.x RunPod hosts report;
# a newer one missing here only narrows capacity. Last checked against RunPod's GPU catalog 2026-10-12.
RUNPOD_ALLOWED_CUDA_VERSIONS = ["13.0", "13.1", "13.2", "13.3", "13.4"]
# Upload-without-GPU mode (cloud.upload_without_gpu): the new pod is stopped straight away and resumed with zero
# GPUs for the upload, then stopped and resumed with its GPU to train. A stopped pod only resumes on its own
# machine, so if someone else took the GPU in between, keep retrying this long before giving up (the pod is then
# left stopped with the upload intact, to start again later).
RUNPOD_GPU_ATTACH_TIMEOUT_SECONDS = 60 * 60
RUNPOD_GPU_ATTACH_RETRY_SECONDS = 60
RUNPOD_STOP_TIMEOUT_SECONDS = 5 * 60
# The pod installs the CUDA 13 torch build (requirements-cuda.txt, cu130). CUDA 13 dropped Maxwell, Pascal
# and Volta (compute capability below 7.5), and CUDA wheels do not run on AMD GPUs at all. Matched against
# the RunPod GPU id / display name; callers only warn on a match, they never refuse the GPU.
_CU130_UNSUPPORTED_GPU = re.compile(
    r"\b(?:V100|GV100|P100|GP100|P40|P6|P4|K80|M40|M60|TITAN V|TITAN Xp?|GTX 10\d0|Quadro P\d+)\b"
    r"|\bAMD\b|\bInstinct\b|\bMI\d{3}",
    re.IGNORECASE,
)


class RunpodGpuUnavailableError(RuntimeError):
    """The stopped pod could not get its GPU back on its machine; the pod and its volume are left intact."""


def runpod_gpu_compat_warning(gpu_type: str, display_name: str = "") -> str | None:
    """A warning when the GPU likely cannot run the pod's cu130 torch build, else None."""
    if not _CU130_UNSUPPORTED_GPU.search(f"{gpu_type} {display_name}"):
        return None
    return (
        f"'{display_name or gpu_type}' is probably incompatible with the pod's CUDA 13 (cu130) torch build: "
        "CUDA 13 dropped GPUs older than Turing (compute capability < 7.5) and does not run on AMD."
    )


class RunpodCloud(LinuxCloud):
    # RunPod pods are always NVIDIA; don't let install.sh detect the platform, since a pod resumed with zero GPUs
    # (upload without GPU) has no nvidia-smi and would get the CPU-only torch build.
    platform_requirements = "requirements-cuda.txt"

    def __init__(self, config: TrainConfig):
        super().__init__(config)

        runpod.api_key = config.secrets.cloud.api_key
        # GPUs to attach when the pod is (re)started; 0 while uploading in upload-without-GPU mode.
        self._gpu_count = 1
        self._wait_for_gpu = False

    def __get_host_port(self):
        secrets = self.config.secrets.cloud
        resumed = False
        while True:
            if (pod := runpod.get_pod(secrets.id)) is None and not resumed:
                raise ValueError(f"Runpod {secrets.id} does not exist")
            if pod and pod["desiredStatus"] == "EXITED":
                self._start()
                # In edge cases runpod returns incorrect information for resumed pods:
                # The pod id seems to disappear for a while, and on recently stopped pods the old public IP and port is still being reported
                # Therefore, on resumed pods, iterate until there is a successful connection
                resumed = True
            elif (
                pod and (runtime := pod["runtime"]) is not None and "ports" in runtime and runtime["ports"] is not None
            ):
                for port in runtime["ports"]:
                    if port["isIpPublic"]:
                        secrets.host = port["ip"]
                        secrets.port = port["publicPort"]
                        if resumed:
                            try:
                                super()._connect()
                            except Exception:
                                continue
                        return
            if secrets.id == "":
                print("waiting for public IP...")
            else:
                print(f"waiting for public IP... Status: https://www.runpod.io/console/pods?id={secrets.id}")
            time.sleep(5)

    def setup(self):
        super().setup()
        self._ensure_remote_rsync()

    def _ensure_remote_rsync(self):
        # rsync-based transfers need the rsync binary on BOTH ends, but the RunPod pytorch base image
        # and the OneTrainer template do not ship it. Install it once (apt + root are available on
        # RunPod) before any upload, so the cache/backup sync does not fail with "command not found".
        if self.config.cloud.file_sync not in (CloudFileSync.NATIVE_RSYNC, CloudFileSync.WSL_RSYNC):
            return
        self.connection.run(
            "command -v rsync >/dev/null 2>&1 "
            "|| (apt-get update && apt-get install -y --no-install-recommends rsync) "
            "|| (echo 'failed to install rsync on the pod' >&2; exit 1)",
            in_stream=False,
        )

    def _connect(self):
        config = self.config.cloud
        secrets = self.config.secrets.cloud

        pod = None
        if secrets.id != "":
            pod = runpod.get_pod(secrets.id)
            if pod is None:
                raise ValueError(f"Runpod {secrets.id} does not exist")
        elif config.create:
            self._create()
            pod = runpod.get_pod(secrets.id)
            if pod is None:
                raise ValueError("Could not create cloud")

        if pod is not None:
            self.__get_host_port()
        super()._connect()

    def _create(self):
        config = self.config.cloud
        secrets = self.config.secrets.cloud
        last_error = None
        pod = None
        print(
            f"Creating RunPod pod: image={RUNPOD_IMAGE}, gpu='{config.gpu_type}', "
            f"type={config.sub_type or 'SECURE'}, persistent volume={config.volume_size}GB @ /workspace",
            flush=True,
        )
        if compat_warning := runpod_gpu_compat_warning(config.gpu_type):
            print(f"  warning: {compat_warning}", flush=True)
        for data_center in RUNPOD_DATA_CENTERS:
            try:
                print(f"  requesting a pod in data center {data_center}...", flush=True)
                pod = runpod.create_pod(
                    name=config.name,
                    image_name=RUNPOD_IMAGE,
                    template_id=RUNPOD_TEMPLATE_ID,
                    gpu_type_id=config.gpu_type,
                    cloud_type=config.sub_type or "SECURE",
                    support_public_ip=True,
                    start_ssh=True,
                    data_center_id=data_center,
                    # Persistent volume (survives a pod stop) holds all the data;
                    # the container disk is ephemeral and kept small.
                    container_disk_in_gb=RUNPOD_CONTAINER_DISK_GB,
                    volume_in_gb=config.volume_size,
                    volume_mount_path="/workspace",
                    allowed_cuda_versions=RUNPOD_ALLOWED_CUDA_VERSIONS,
                    min_download=config.min_download or None,
                    env={"JUPYTER_PASSWORD": pysecrets.token_urlsafe(16)},
                )
                # runpod returns None (rather than raising) when a data center has no capacity for
                # the requested GPU; keep trying the remaining ones instead of bailing on the first miss.
                if pod:
                    print(f"  pod created in {data_center}: id={pod.get('id')}", flush=True)
                    break
                print(f"  no capacity for '{config.gpu_type}' in {data_center}", flush=True)
            except Exception as exc:
                print(f"  {data_center} failed: {exc}", flush=True)
                last_error = exc

        if not pod:
            raise last_error or RuntimeError(
                f"Could not create a '{config.gpu_type}' pod in any of {RUNPOD_DATA_CENTERS} (no capacity). "
                f"Check availability/region in the RunPod console, or edit RUNPOD_DATA_CENTERS."
            )
        secrets.id = pod["id"]

        if config.upload_without_gpu:
            # Stop the pod right away; _connect() then finds it EXITED and _start() resumes it with zero GPUs,
            # so the upload is billed at the CPU-only rate instead of the GPU's.
            print("upload without GPU: stopping the new pod so the upload runs with zero GPUs attached...", flush=True)
            self._stop_and_wait()
            self._gpu_count = 0

    def attach_gpu(self):
        """After an upload without GPU: stop the pod, resume it on the same machine with its GPU and reconnect."""
        if self._gpu_count != 0:
            return
        secrets = self.config.secrets.cloud
        print("upload finished: stopping the pod to resume it with its GPU...", flush=True)
        self.close()
        self._stop_and_wait()
        self._gpu_count = 1
        self._wait_for_gpu = True
        secrets.host = ""
        secrets.port = ""
        self._connect()
        # the container disk is wiped on a stop: anything installed outside /workspace is gone again
        self.connection.run(f"mkfifo {shlex.quote(self.command_pipe)}", warn=True, hide=True, in_stream=False)
        self._ensure_remote_rsync()
        if self.config.cloud.tensorboard_tunnel:
            self._make_tensorboard_tunnel()
        print("GPU attached; continuing.", flush=True)

    def _stop_and_wait(self):
        pod_id = self.config.secrets.cloud.id
        runpod.stop_pod(pod_id)
        deadline = time.monotonic() + RUNPOD_STOP_TIMEOUT_SECONDS
        while (pod := runpod.get_pod(pod_id)) is not None and pod.get("desiredStatus") != "EXITED":
            if time.monotonic() >= deadline:
                raise RuntimeError(f"RunPod pod {pod_id} did not stop within {RUNPOD_STOP_TIMEOUT_SECONDS} seconds")
            time.sleep(5)

    def delete(self):
        runpod.terminate_pod(self.config.secrets.cloud.id)

    def stop(self):
        runpod.stop_pod(self.config.secrets.cloud.id)

    def _start(self):
        pod_id = self.config.secrets.cloud.id
        if self._gpu_count == 0:
            try:
                runpod.resume_pod(pod_id, gpu_count=0)
                print("pod resumed with zero GPUs for the upload", flush=True)
                return
            except Exception as exc:
                # don't lose the pod over it: upload with the GPU attached, as without upload_without_gpu
                print(f"could not resume the pod without a GPU ({exc}); uploading with the GPU attached", flush=True)
                self._gpu_count = 1
        if not self._wait_for_gpu:
            runpod.resume_pod(pod_id, gpu_count=1)
            return
        deadline = time.monotonic() + RUNPOD_GPU_ATTACH_TIMEOUT_SECONDS
        while True:
            try:
                runpod.resume_pod(pod_id, gpu_count=1)
                return
            except Exception as exc:  # noqa: PERF203 — the retry is the point
                if time.monotonic() >= deadline:
                    raise RunpodGpuUnavailableError(
                        f"RunPod pod {pod_id} could not get its GPU back within "
                        f"{RUNPOD_GPU_ATTACH_TIMEOUT_SECONDS // 60} minutes ({exc}). The pod is left stopped with the "
                        "upload on its volume: start training again later to retry, or migrate the pod in the RunPod "
                        "console."
                    ) from exc
                print(
                    f"GPU not free on the pod's machine yet ({exc}); retrying in {RUNPOD_GPU_ATTACH_RETRY_SECONDS}s...",
                    flush=True,
                )
                time.sleep(RUNPOD_GPU_ATTACH_RETRY_SECONDS)

    def _get_action_cmd(self, action: CloudAction):
        if action == CloudAction.STOP:
            return "source /etc/rp_environment && runpodctl stop pod $RUNPOD_POD_ID"
        elif action == CloudAction.DELETE:
            return "source /etc/rp_environment && runpodctl remove pod $RUNPOD_POD_ID"
        else:
            return ":"

    def _get_detached_watchdog_cmd(self) -> str:
        stop_cmd = self._get_action_cmd(CloudAction.STOP)
        script = f"""
pid_file={shlex.quote(self.pid_file)}
missing_gpu_seconds=0
echo "RunPod watchdog started: stopping pod if trainer exits or nvidia-smi is unavailable for 1800 seconds."
while true; do
    sleep 60
    if [ ! -s "$pid_file" ]; then
        echo "RunPod watchdog: pid file missing or empty; stopping pod."
        {stop_cmd}
        exit 0
    fi
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        echo "RunPod watchdog: trainer process is no longer running; stopping pod."
        {stop_cmd}
        exit 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        missing_gpu_seconds=0
    else
        missing_gpu_seconds=$((missing_gpu_seconds + 60))
        echo "RunPod watchdog: nvidia-smi unavailable for $missing_gpu_seconds seconds."
        if [ "$missing_gpu_seconds" -ge 1800 ]; then
            echo "RunPod watchdog: GPU status unavailable for 1800 seconds; stopping pod."
            {stop_cmd}
            exit 0
        fi
    fi
done
""".strip()
        return f"(nohup sh -c {shlex.quote(script)} >> {shlex.quote(self.log_file)} 2>&1 &)"
