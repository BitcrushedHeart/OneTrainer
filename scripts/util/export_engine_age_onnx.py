"""
Export Bitcrush Engine's Age group checkpoint to the ONNX + sidecar the DPO
tool's age estimator reads (``external/models/age/age_base.onnx`` + ``.json``).

Run with the **Bitcrush Suite** environment, from ``apps/engine/src/backend``,
because it needs BitTrainer (the model definition), the shared ONNX exporter
and Engine's checkpoint-metadata readers::

    cd "F:/Projects/Bitcrush/Bitcrush Suite/apps/engine/src/backend"
    CUDA_VISIBLE_DEVICES=-1 "../../../../.venv/Scripts/python.exe" \\
        "E:/AI/Data/Packages/OneTrainer/scripts/util/export_engine_age_onnx.py" \\
        --checkpoint "F:/Projects/Bitcrush/Bitcrush Suite/apps/engine/models/BMC_Age.pt" \\
        --model-version cea822ceb4b82df6 \\
        --dest "E:/AI/Data/Packages/OneTrainer/external/models/age/age_base.onnx"

What gets baked into the exported classifier, so the consumer only softmaxes:
the Bayes prior correction (``tau * (log natural - log effective)``), the
softmax temperature and the per-class logit bias — in Engine's own order
(prior first, then calibration). ``--model-version`` is the group's
``model_version`` in Engine's DB; caches key on it, so a retrained head must
be exported under a new version.

The script checks torch/ONNX parity on ``--probe-image`` (any photo) and
refuses to write the sidecar if they disagree.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dest", required=True, help="output .onnx path; the .json sidecar sits beside it")
    parser.add_argument("--model-version", required=True, help="Engine groups.model_version for this checkpoint")
    parser.add_argument("--model-size", default="base")
    parser.add_argument("--num-classes", type=int, default=102)
    parser.add_argument("--probe-image", default=None, help="image for the torch/ONNX parity check")
    parser.add_argument("--parity-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    import torch

    import numpy as np
    from backend.services.group_predict import (
        _load_ordinal_decode,
        _load_prior_delta,
        _load_softmax_calibration,
    )
    from bitcrush_pyshared.inference import bucket_preprocess as bp
    from bitcrush_pyshared.inference import export_classifier_onnx
    from bittrainer.model import load_checkpoint

    ckpt = args.checkpoint
    data = torch.load(ckpt, map_location="cpu", weights_only=True)
    delta = _load_prior_delta(ckpt)
    temperature, bias = _load_softmax_calibration(ckpt)
    cut_points, none_index = _load_ordinal_decode(ckpt)

    model = load_checkpoint(
        ckpt, device="cpu", dtype=torch.float32, model_size=args.model_size, num_classes=args.num_classes
    ).eval()
    baked = copy.deepcopy(model)
    head = baked.get_classifier()
    with torch.no_grad():
        b = head.bias.clone() if head.bias is not None else torch.zeros(head.weight.shape[0])
        if delta:
            b = b + torch.tensor(delta, dtype=torch.float32)
        head.weight.div_(float(temperature))
        b = b / float(temperature)
        if bias:
            b = b + torch.tensor(bias, dtype=torch.float32)
        if head.bias is None:
            head.bias = torch.nn.Parameter(b)
        else:
            head.bias.copy_(b)

    os.makedirs(os.path.dirname(os.path.abspath(args.dest)), exist_ok=True)
    export_classifier_onnx(baked.eval(), args.dest)

    if args.probe_image:
        import onnxruntime as ort
        from PIL import Image

        im = Image.open(args.probe_image).convert("RGB")
        bw, bh = bp.find_nearest_bucket(*im.size)
        im = bp.crop_to_bucket_aspect(im, bw / bh).resize((bw, bh), Image.LANCZOS)
        arr = (
            (np.asarray(im, dtype=np.float32) / 255.0 - np.array(bp.IMAGENET_MEAN, dtype=np.float32))
            / np.array(bp.IMAGENET_STD, dtype=np.float32)
        ).transpose(2, 0, 1)[None]
        with torch.no_grad():
            torch_logits = baked(torch.from_numpy(arr)).numpy()
        session = ort.InferenceSession(args.dest, providers=["CPUExecutionProvider"])
        onnx_logits = session.run(["logits"], {"pixels": arr})[0]
        diff = float(np.abs(torch_logits - onnx_logits).max())
        print(f"parity max|diff| = {diff:.2e}")
        if diff > args.parity_tolerance:
            print("ONNX export does not match torch; sidecar not written", file=sys.stderr)
            return 1

    sidecar = {
        "model_size": args.model_size,
        "model_version": args.model_version,
        "source_checkpoint": os.path.basename(ckpt),
        "train_resolution": bp.DEFAULT_TRAIN_RESOLUTION,
        "aspect_ratio_buckets": [list(bucket) for bucket in bp.scaled_buckets(bp.DEFAULT_TRAIN_RESOLUTION)],
        "imagenet_mean": list(bp.IMAGENET_MEAN),
        "imagenet_std": list(bp.IMAGENET_STD),
        "class_names": data.get("class_names"),
        "none_index": none_index,
        "min_age": 0,
        "ordinal_cut_points": cut_points,
        "baked": {
            "prior_delta": bool(delta),
            "temperature": float(temperature),
            "class_logit_bias": bool(bias),
        },
        "preprocess": (
            "convert RGB -> centre-crop to nearest bucket aspect -> LANCZOS resize to bucket dims "
            "-> /255 -> imagenet normalise -> CHW; softmax(logits) is already prior-corrected and calibrated"
        ),
    }
    sidecar_path = os.path.splitext(args.dest)[0] + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)
    print(f"wrote {args.dest} ({os.path.getsize(args.dest) / 1e6:.0f} MB) and {sidecar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
