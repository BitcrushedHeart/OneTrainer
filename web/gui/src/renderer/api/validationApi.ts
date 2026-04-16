import { request } from "./request";

export interface ValidationMatch {
    id: string;
    train_image: string;
    train_path: string;
    train_image_url?: string;
    val_image: string;
    val_path: string;
    val_image_url?: string;
    kind: string;
    detail: string;
    score: number | null;
}

export interface ValidationResults {
    matches?: ValidationMatch[];
    captions?: Array<{
        caption: string;
        train_images: string[];
        val_images: string[];
    }>;
    summary?: {
        hash_matches: number;
        perceptual_matches: number;
        filename_matches: number;
        caption_groups: number;
        train_images: number;
        val_images: number;
        clip_matches?: number;
    };
    error?: string;
}

export interface ActionResponse {
    ok: boolean;
    error?: string;
}

export const validationApi = {
    getStatus: () =>
        request<{ status: string; progress: { label: string; current: number; total: number } }>("/validation/status"),

    getResults: () => request<ValidationResults>("/validation/results"),

    scanBasic: () => request<ActionResponse>("/validation/scan/basic", { method: "POST" }),

    scanDeep: (threshold = 0.93) =>
        request<ActionResponse>("/validation/scan/deep", {
            method: "POST",
            body: JSON.stringify({ threshold }),
        }),

    cancel: () => request<ActionResponse>("/validation/cancel", { method: "POST" }),

    removeMatch: (valImagePath: string) =>
        request<ActionResponse>("/validation/match/remove", {
            method: "POST",
            body: JSON.stringify({ val_image_path: valImagePath }),
        }),

    moveMatch: (valImagePath: string) =>
        request<{ ok: boolean; new_path?: string }>("/validation/match/move", {
            method: "POST",
            body: JSON.stringify({ val_image_path: valImagePath }),
        }),

    removeMatchBatch: (valImagePaths: string[]) =>
        request<{ ok: boolean; removed: number; errors: string[] }>("/validation/match/remove-batch", {
            method: "POST",
            body: JSON.stringify({ val_image_paths: valImagePaths }),
        }),
};
