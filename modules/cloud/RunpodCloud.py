import secrets as pysecrets
import shlex
import time

from modules.cloud.LinuxCloud import LinuxCloud
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.CloudAction import CloudAction
from modules.util.enum.CloudFileSync import CloudFileSync

import runpod

RUNPOD_TEMPLATE_ID = "1a33vbssq9"
EUROPE_COUNTRY_CODES = ["EU", "GB", "NL", "DE", "FR", "SE", "NO", "FI", "PL"]


class RunpodCloud(LinuxCloud):
    def __init__(self, config: TrainConfig):
        super().__init__(config)

        runpod.api_key = config.secrets.cloud.api_key

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
        for country_code in EUROPE_COUNTRY_CODES:
            try:
                pod = runpod.create_pod(
                    name=config.name,
                    image_name="",
                    template_id=RUNPOD_TEMPLATE_ID,
                    gpu_type_id=config.gpu_type,
                    cloud_type=config.sub_type or "SECURE",
                    support_public_ip=True,
                    start_ssh=True,
                    country_code=country_code,
                    volume_in_gb=config.volume_size,
                    volume_mount_path="/workspace",
                    min_download=config.min_download or None,
                    env={"JUPYTER_PASSWORD": pysecrets.token_urlsafe(16)},
                )
                # runpod returns None (rather than raising) when a region has no capacity for the
                # requested GPU; keep trying the remaining countries instead of bailing on the
                # first miss.
                if pod:
                    break
            except Exception as exc:
                last_error = exc

        if not pod:
            raise last_error or RuntimeError(
                f"Could not create a '{config.gpu_type}' pod in any European region (no capacity)"
            )
        secrets.id = pod["id"]

    def delete(self):
        runpod.terminate_pod(self.config.secrets.cloud.id)

    def stop(self):
        runpod.stop_pod(self.config.secrets.cloud.id)

    def _start(self):
        runpod.resume_pod(self.config.secrets.cloud.id, gpu_count=1)

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
