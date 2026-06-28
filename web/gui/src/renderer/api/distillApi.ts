import { request } from "@/api/request";

export interface DistillBuildFilters {
  model_contains?: string | null;
  steps?: number | null;
  cfg_scale?: number | null;
  scheduler?: string | null;
  width?: number | null;
  height?: number | null;
}

export interface DistillBuildDatasetRequest {
  source_folder: string;
  output_folder: string;
  include_subdirectories: boolean;
  val_percentage: number;
  filters: DistillBuildFilters;
}

export interface DistillManifest {
  ok: boolean;
  source_folder: string;
  output_folder: string;
  scanned: number;
  accepted: number;
  train: number;
  val: number;
  rejected: Record<string, number>;
  items: Array<Record<string, unknown>>;
}

export const distillApi = {
  buildDataset: (params: DistillBuildDatasetRequest) =>
    request<DistillManifest>("/distill/build-dataset", {
      method: "POST",
      body: JSON.stringify(params),
    }),
};
