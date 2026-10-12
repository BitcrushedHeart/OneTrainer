import { request } from "@/api/request";
import type { TrainConfig } from "@/types/generated/config";

export interface RunpodSizeEntry {
  label: string;
  path: string;
  exists: boolean;
  bytes: number;
  gib: number;
  files: number;
  required: boolean;
}

export interface RunpodPreview {
  entries: RunpodSizeEntry[];
  buffer_gib: number;
  total_gib: number;
  required_gb: number;
  latest_backup: string;
  remote_paths: Record<string, string>;
  errors: string[];
  warnings: string[];
  git: {
    main_dirty: boolean;
    mgds_dirty: boolean;
    mgds_path: string;
    requirements_path: string;
  };
  defaults: {
    template_id: string;
    gpu_type: string;
    cloud_type: string;
    cloud_types?: string[];
    install_cmd: string;
  };
}

export interface RunpodGpuType {
  id: string;
  display_name: string;
  memory_gb: number | null;
  secure_cloud: boolean | null;
  community_cloud: boolean | null;
  secure_price: number | null;
  community_price: number | null;
  /** Set when the GPU likely cannot run the pod's cu130 torch build. Informational only. */
  compat_warning: string | null;
}

export interface RunpodGpuList {
  ok: boolean;
  gpus: RunpodGpuType[];
  default: string;
  error: string | null;
}

export interface RunpodGitPreflight {
  ok: boolean;
  mgds: {
    dirty: boolean;
    commit: string;
    branch: string;
  };
  requirements_changed: boolean;
  main: {
    dirty: boolean;
    commit: string;
    branch: string;
    remote: string;
  };
}

export interface RunpodPrepareRequest {
  api_key: string;
  ssh_user: string;
  ssh_key_file: string;
  ssh_password: string;
  pod_name: string;
  run_id: string;
  min_download: number;
  start: boolean;
  epochs?: number;
  /** RunPod GPU type id; empty keeps the backend default (RTX 5090). */
  gpu_type?: string;
  cloud_type?: string;
}

export interface RunpodPrepareResponse {
  ok: boolean;
  config: TrainConfig;
  preview: RunpodPreview;
  warnings?: string[];
  start?: {
    ok: boolean;
    error?: string | null;
  };
}

export interface RunpodLiveTestResponse {
  ok: boolean;
  preview: RunpodPreview;
  smoke: {
    cache_dir: string;
    workspace_dir: string;
    image_path: string;
    text_path: string;
    epochs: number;
  };
  start: {
    ok: boolean;
    error?: string | null;
  };
}

export const cloudApi = {
  runpodPreview: () => request<RunpodPreview>("/cloud/runpod/preview"),

  runpodGpus: (apiKey: string) =>
    request<RunpodGpuList>("/cloud/runpod/gpus", {
      method: "POST",
      body: JSON.stringify({ api_key: apiKey }),
    }),

  runpodGitPreflight: () =>
    request<RunpodGitPreflight>("/cloud/runpod/git-preflight", {
      method: "POST",
    }),

  runpodPrepare: (body: RunpodPrepareRequest) =>
    request<RunpodPrepareResponse>("/cloud/runpod/prepare", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  runpodLiveTest: (body: RunpodPrepareRequest) =>
    request<RunpodLiveTestResponse>("/cloud/runpod/live-test", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
