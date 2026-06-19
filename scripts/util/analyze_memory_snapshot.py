"""
Analyze a torch.cuda memory snapshot (.pickle from torch.cuda.memory._dump_snapshot)
and print a per-callsite breakdown of where VRAM is being held.

Usage:
    python scripts/util/analyze_memory_snapshot.py memory-step0.pickle
    python scripts/util/analyze_memory_snapshot.py memory-step0.pickle --top 40
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

# Frames in these path fragments are "plumbing" -- skip past them to find the
# user-level call site that actually requested the memory.
PLUMBING_PATTERNS = [
    r"site-packages[\\/]torch[\\/]nn[\\/]modules[\\/](module|container|activation)",
    r"site-packages[\\/]torch[\\/]_dynamo",
    r"site-packages[\\/]torch[\\/]autograd[\\/]__init__",
    r"site-packages[\\/]torch[\\/]_tensor",
    r"site-packages[\\/]torch[\\/]utils[\\/]checkpoint",
    r"site-packages[\\/]torch[\\/]overrides",
    r"site-packages[\\/]torch[\\/]_compile",
    r"threading\.py$",
    r"web[\\/]backend[\\/]services[\\/]trainer_service",
    r"modules[\\/]trainer[\\/]GenericTrainer\.py$",
]
_PLUMBING_RE = re.compile("|".join(PLUMBING_PATTERNS))


def is_plumbing(frame: dict) -> bool:
    return bool(_PLUMBING_RE.search(frame.get("filename", "")))


def short_path(path: str) -> str:
    # Strip everything up to and including site-packages/ or src/ or the repo root
    for marker in ("site-packages\\", "site-packages/", "src\\", "src/", "OneTrainer\\", "OneTrainer/"):
        idx = path.rfind(marker)
        if idx != -1:
            return path[idx + len(marker) :]
    return path


def pick_attribution_frame(frames: list[dict]) -> dict | None:
    """Return the deepest (top-of-stack) frame that isn't pure plumbing."""
    if not frames:
        return None
    for fr in frames:
        if not is_plumbing(fr):
            return fr
    return frames[0]


# Coarse category buckets keyed by patterns in the attribution frame's filename
CATEGORY_RULES = [
    (r"modules[\\/]util[\\/]optimizer", "optimizer"),
    (r"site-packages[\\/]torch[\\/]optim", "optimizer"),
    (r"bitsandbytes", "optimizer"),
    (r"modules[\\/]util[\\/]checkpointing_util", "activations (checkpoint)"),
    (r"modules[\\/]util[\\/]LayerOffload", "offload conductor"),
    (r"diffusers[\\/]models[\\/]transformers", "transformer activations"),
    (r"diffusers[\\/]models[\\/]normalization", "transformer activations"),
    (r"diffusers[\\/]models[\\/]attention", "transformer activations"),
    (r"diffusers[\\/]models[\\/]autoencoders", "vae"),
    (r"transformers[\\/]models", "text encoder"),
    (r"modules[\\/]model[\\/].*Model\.py", "model setup / encode"),
    (r"modules[\\/]modelSetup", "model setup / encode"),
    (r"modules[\\/]dataLoader", "dataloader"),
    (r"site-packages[\\/]torch[\\/]nn[\\/]functional", "torch.nn.functional"),
    (r"site-packages[\\/]torch[\\/]nn[\\/]modules[\\/]linear", "linear forward"),
    (r"site-packages[\\/]torch[\\/]nn[\\/]modules[\\/]conv", "conv forward"),
    (r"site-packages[\\/]torch[\\/]nn[\\/]modules[\\/]normalization", "norm forward"),
    (r"site-packages[\\/]torch[\\/]autograd", "autograd / backward"),
]


def categorize(frame: dict | None) -> str:
    if frame is None:
        return "<unknown / pre-recording>"
    fn = frame.get("filename", "")
    for pat, label in CATEGORY_RULES:
        if re.search(pat, fn):
            return label
    return "other"


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):8.3f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):8.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):8.1f} KiB"
    return f"{n:8d} B"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("snapshot", help="path to memory snapshot pickle")
    ap.add_argument("--top", type=int, default=30, help="top-N call sites to show")
    ap.add_argument(
        "--min-mib", type=float, default=1.0, help="omit call sites smaller than this many MiB from the per-site table"
    )
    args = ap.parse_args()

    with open(args.snapshot, "rb") as f:
        snap = pickle.load(f)

    segments = snap["segments"]
    traces = snap.get("device_traces", [[]])
    traces = traces[0] if traces else []

    # ---- aggregate segment-level totals -------------------------------------
    total_reserved = sum(s["total_size"] for s in segments)
    total_allocated = sum(s["allocated_size"] for s in segments)
    total_active = sum(s["active_size"] for s in segments)
    total_requested = sum(s["requested_size"] for s in segments)

    free_in_segments = total_reserved - total_allocated  # internal fragmentation
    rounding_overhead = total_allocated - total_requested

    print("=" * 78)
    print(f"Snapshot: {args.snapshot}  ({os.path.getsize(args.snapshot) / 1e6:.1f} MB on disk)")
    print("=" * 78)
    print(f"  Reserved by allocator (cudaMalloc'd):  {fmt_bytes(total_reserved)}")
    print(f"  Allocated to live blocks:              {fmt_bytes(total_allocated)}")
    print(f"  Requested by user code:                {fmt_bytes(total_requested)}")
    print(f"  Free inside reserved segments (frag):  {fmt_bytes(free_in_segments)}")
    print(f"  Rounding overhead (alloc - requested): {fmt_bytes(rounding_overhead)}")
    print(f"  Segments: {len(segments):,}    device trace events: {len(traces):,}")
    print()

    # ---- stitch frames onto live blocks via device_traces -------------------
    # For each address, remember the most recent alloc-event frames.
    addr_to_frames: dict[int, list[dict]] = {}
    for t in traces:
        if t.get("action") == "alloc" and t.get("frames"):
            addr_to_frames[t["addr"]] = t["frames"]

    # ---- walk live blocks and attribute -------------------------------------
    per_site_bytes: dict[tuple, int] = defaultdict(int)
    per_site_count: dict[tuple, int] = defaultdict(int)
    per_category_bytes: dict[str, int] = defaultdict(int)
    per_category_count: dict[str, int] = defaultdict(int)
    unattributed_bytes = 0
    unattributed_count = 0

    for seg in segments:
        for blk in seg["blocks"]:
            if not blk["state"].startswith("active"):
                continue  # skip free blocks
            frames = blk.get("frames") or addr_to_frames.get(blk["address"]) or []
            attribution = pick_attribution_frame(frames)
            category = categorize(attribution)
            per_category_bytes[category] += blk["size"]
            per_category_count[category] += 1
            if attribution is None:
                unattributed_bytes += blk["size"]
                unattributed_count += 1
                continue
            key = (attribution["filename"], attribution["line"], attribution["name"])
            per_site_bytes[key] += blk["size"]
            per_site_count[key] += 1

    attributed_total = sum(per_site_bytes.values())

    # ---- categories -----------------------------------------------------------
    print("By category (live blocks):")
    print("-" * 78)
    print(f"  {'category':35s} {'bytes':>14s} {'%':>6s} {'count':>10s}")
    for cat, bytes_ in sorted(per_category_bytes.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * bytes_ / total_allocated if total_allocated else 0
        print(f"  {cat:35s} {fmt_bytes(bytes_):>14s} {pct:5.1f}% {per_category_count[cat]:10,d}")
    if unattributed_count:
        pct = 100.0 * unattributed_bytes / total_allocated if total_allocated else 0
        print(f"  {'(unattributed)':35s} {fmt_bytes(unattributed_bytes):>14s} {pct:5.1f}% {unattributed_count:10,d}")
    print()

    # ---- top call sites -------------------------------------------------------
    print(f"Top {args.top} call sites (>= {args.min_mib} MiB), by total live bytes:")
    print("-" * 78)
    min_bytes = int(args.min_mib * (1 << 20))
    sorted_sites = sorted(per_site_bytes.items(), key=lambda kv: -kv[1])
    shown = 0
    for (fn, line, name), bytes_ in sorted_sites:
        if bytes_ < min_bytes:
            break
        if shown >= args.top:
            break
        pct = 100.0 * bytes_ / total_allocated if total_allocated else 0
        count = per_site_count[(fn, line, name)]
        avg = bytes_ // count if count else 0
        print(
            f"  {fmt_bytes(bytes_):>11s} {pct:5.1f}%  n={count:5d}  avg={fmt_bytes(avg):>11s}  "
            f"{name}  {short_path(fn)}:{line}"
        )
        shown += 1
    print()

    # ---- peak memory replay --------------------------------------------------
    if traces:
        live = 0
        peak = 0
        peak_idx = -1
        addr_size: dict[int, int] = {}
        for i, t in enumerate(traces):
            action = t.get("action")
            if action == "alloc":
                addr_size[t["addr"]] = t["size"]
                live += t["size"]
                if live > peak:
                    peak = live
                    peak_idx = i
            elif action == "free_completed":
                sz = addr_size.pop(t["addr"], 0)
                live -= sz
        print(f"Peak live memory during recording: {fmt_bytes(peak)} (at trace #{peak_idx} of {len(traces)})")
        if peak_idx >= 0:
            t = traces[peak_idx]
            print(f"  trigger: {t.get('action')} addr={hex(t['addr'])} size={fmt_bytes(t['size'])}")
            for fr in t.get("frames", [])[:8]:
                print(f"    at {fr['name']}  {short_path(fr['filename'])}:{fr['line']}")
    print()

    # ---- allocator hints -----------------------------------------------------
    settings = snap.get("allocator_settings", {})
    if settings:
        print("Allocator settings:")
        for k, v in settings.items():
            print(f"  {k}: {v}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
