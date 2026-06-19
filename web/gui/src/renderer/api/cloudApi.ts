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
    install_cmd: string;
  };
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
}

export interface RunpodPrepareResponse {
  ok: boolean;
  config: TrainConfig;
  preview: RunpodPreview;
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
