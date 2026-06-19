"""Manually walk the failing item through the upstream pipeline modules,
logging the shape at each stage. No model, no cache — just the pure
augmentation/bucketing chain."""

import sys

from torchvision import transforms

from PIL import Image

IMG = r"F:\Concepts\Braless\train\66215583_013_fdae.jpg"
MASK = r"F:\Concepts\Braless\train\66215583_013_fdae-masklabel.png"

from random import Random

from mgds.pipelineModules.AspectBucketing import AspectBucketing

import numpy as np


def main():
    img_pil = Image.open(IMG).convert("RGB")
    msk_pil = Image.open(MASK).convert("L")
    print(f"image file: {img_pil.size} (W x H)  →  tensor (C, H, W) = (3, {img_pil.size[1]}, {img_pil.size[0]})")
    print(f"mask file:  {msk_pil.size}  →  tensor (1, {msk_pil.size[1]}, {msk_pil.size[0]})")

    # CalcAspect produces resolution = tuple(image.shape[1:])  → (H, W)
    image_t = transforms.functional.to_tensor(img_pil)
    mask_t = transforms.functional.to_tensor(msk_pil)
    original_resolution = tuple(image_t.shape[1:])
    print(f"\nCalcAspect output: original_resolution={original_resolution}")
    print(f"  aspect (h/w) = {original_resolution[0] / original_resolution[1]:.4f}")

    for targets in [[512], [512, 768], [768], [512, 768, 1024]]:
        ab = AspectBucketing.__new__(AspectBucketing)
        ab.quantization = 64
        ab.bucket_resolutions, ab.bucket_aspects = ab._AspectBucketing__create_automatic_buckets(targets)
        ab.bucket_aspects = {k: np.array(v) for k, v in ab.bucket_aspects.items()}

        for variation in range(2):
            for index in [0, 1, 100, 12345]:
                seed = hash((42, 7, variation, index))
                rand = Random(seed)
                target_resolutions = list(targets)
                target_resolution = rand.choice(target_resolutions)
                bucket = ab.bucket_for_aspect(original_resolution[0] / original_resolution[1], target_resolution)
                aspect = original_resolution[0] / original_resolution[1]
                target_aspect = bucket[0] / bucket[1]
                if aspect > target_aspect:
                    scale = bucket[1] / original_resolution[1]
                    scale_res = (round(original_resolution[0] * scale), bucket[1])
                else:
                    scale = bucket[0] / original_resolution[0]
                    scale_res = (bucket[0], round(original_resolution[1] * scale))
                latent_h = bucket[0] // 8
                latent_w = bucket[1] // 8
                print(
                    f"  targets={targets} v={variation} idx={index}: target_res={target_resolution} crop_res={bucket} scale_res={scale_res} latent_mask_shape=(1, {latent_h}, {latent_w})"
                )
        print()


if __name__ == "__main__":
    sys.exit(main())
