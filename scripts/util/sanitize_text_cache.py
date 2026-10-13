"""Strip caption text out of OneTrainer text caches (SmartDiskCache dirs).

Captions live in two places in a text cache:

* ``cache.json``: ``prompt`` / ``prompt_1`` / ``prompt_2`` strings in every
  row's ``sourceless_rows[*].runtime_values[*]`` block (and the legacy
  entry-level ``sourceless_runtime_values[*]`` block). They are set to ``""``;
  the keys and every other field are kept so sourceless validation still sees
  a fully stamped row.
* every ``.pt``: the token-id tensors (``tokens`` by default) are refilled with
  a constant (zeros, matching mgds ``scrub_split_names``; or the tokenizer's
  pad id with ``--fill pad``), and the legacy embedded
  ``__sourceless_values`` prompt strings are blanked. Hidden states and
  attention masks are untouched.

Nothing in this script prints caption text or token ids: progress and the
verification report are key names, types, shapes and counts only.

Modes
-----
Staging copy (originals are never written; a size+mtime snapshot of every
source file, plus sha256 of the index files and a .pt sample, is compared
before/after)::

    python scripts/util/sanitize_text_cache.py SRC [SRC ...] --staging-root DIR

Each SRC (e.g. ``...\\Sourceless\\text``) is written to
``DIR\\<parent name>\\<name>``. Re-running resumes: outputs whose mtime matches
their source are skipped.

In place (backs up only ``cache.json`` as ``cache.json.pre-scrub-<stamp>.bak``
beside it; re-running is a no-op for already scrubbed files)::

    python scripts/util/sanitize_text_cache.py SRC [SRC ...] --in-place --confirm-in-place

``--verify-only`` re-runs the verification of a previous run without writing.
``--functional-check N`` builds a read-only sourceless image+text pipeline over
the result and loads N rows through SmartDiskCache.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter

PROMPT_KEYS = ("prompt", "prompt_1", "prompt_2")
DEFAULT_SCRUB_NAMES = ("tokens",)
# Non-tensor .pt keys whose string values are known not to be caption text.
KNOWN_PT_STRING_KEYS = {"__modeltype", "__content_hash", "__concept_type", "__concept_name", "__concept_path"}
TMP_SUFFIX = ".sanitize.tmp"

# ---------------------------------------------------------------------------
# pad id (tokenizer metadata only; nothing is decoded)
# ---------------------------------------------------------------------------


def find_pad_id(tokenizer_dir: str | None) -> tuple[int, str]:
    if tokenizer_dir:
        configs = [os.path.join(tokenizer_dir, "tokenizer_config.json")]
    else:
        hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        configs = sorted(
            glob.glob(
                os.path.join(
                    hub, "models--Tongyi-MAI--Z-Image*", "snapshots", "*", "tokenizer", "tokenizer_config.json"
                )
            )
        )
    ids = {}
    for path in configs:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        pad = cfg.get("pad_token")
        if isinstance(pad, dict):
            pad = pad.get("content")
        for tid, tok in (cfg.get("added_tokens_decoder") or {}).items():
            if tok.get("content") == pad:
                ids[path] = int(tid)
    if not ids or len(set(ids.values())) != 1:
        raise SystemExit(f"could not resolve a unique pad id from {configs}: {ids}")
    path, pad_id = next(iter(ids.items()))
    return pad_id, path


# ---------------------------------------------------------------------------
# cache.json
# ---------------------------------------------------------------------------

# Containers whose keys are data (file paths, hashes, indices), not schema.
DYNAMIC_CONTAINERS = {
    "entries",
    "hash_index",
    "concepts",
    "sourceless_rows",
    "runtime_values",
    "sourceless_runtime_values",
    "variants",
    "roles",
    "validations",
    "sidecar_mtimes",
    "sidecar_hashes",
    "watched_fingerprints",
}
_SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")


def _pattern_key(parent: str | None, key) -> str:
    if parent in DYNAMIC_CONTAINERS:
        return "<*>"
    key = str(key)
    if _SAFE_KEY.match(key) and not re.fullmatch(r"[0-9a-f]{8,}", key):
        return key
    return "<*>"


def blank_prompts(node, counts: Counter, path: str = "", parent: str | None = None):
    """Set every prompt/prompt_1/prompt_2 string to "" (in place)."""
    if isinstance(node, dict):
        for key in list(node):
            pk = _pattern_key(parent, key)
            value = node[key]
            if key in PROMPT_KEYS and parent not in DYNAMIC_CONTAINERS:
                loc = f"{path}/{pk}"
                if isinstance(value, str):
                    counts[f"{loc} (str, non-empty blanked)" if value else f"{loc} (str, already empty)"] += 1
                    node[key] = ""
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    counts[f"{loc} (list[str] blanked)"] += 1
                    node[key] = ["" for _ in value]
                else:
                    counts[f"{loc} (UNHANDLED {type(value).__name__})"] += 1
                continue
            blank_prompts(value, counts, f"{path}/{pk}", str(key) if parent not in DYNAMIC_CONTAINERS else pk)
    elif isinstance(node, list):
        for item in node:
            blank_prompts(item, counts, f"{path}[]", parent)


def string_leaf_report(node, out: dict, path: str = "", parent: str | None = None):
    """pattern -> [total str leaves, non-empty str leaves]. Paths only, never values."""
    if isinstance(node, dict):
        for key, value in node.items():
            pk = _pattern_key(parent, key)
            string_leaf_report(value, out, f"{path}/{pk}", str(key) if parent not in DYNAMIC_CONTAINERS else pk)
    elif isinstance(node, list):
        for item in node:
            string_leaf_report(item, out, f"{path}[]", parent)
    elif isinstance(node, str):
        slot = out.setdefault(path, [0, 0])
        slot[0] += 1
        slot[1] += bool(node)


def count_nonempty_prompts(node, parent: str | None = None) -> int:
    n = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if key in PROMPT_KEYS and parent not in DYNAMIC_CONTAINERS:
                if isinstance(value, str):
                    n += bool(value)
                elif isinstance(value, list):
                    n += sum(bool(v) for v in value)
                else:
                    n += 1  # unknown type counts as not clean
                continue
            n += count_nonempty_prompts(value, str(key) if parent not in DYNAMIC_CONTAINERS else "<*>")
    elif isinstance(node, list):
        for item in node:
            n += count_nonempty_prompts(item, parent)
    return n


def dump_index(index: dict) -> bytes:
    # Same as SmartDiskCache._save_cache_index(compact=True).
    return json.dumps(index, separators=(",", ":")).encode("utf-8")


def atomic_write_bytes(path: str, payload: bytes, mtime_ns: int | None = None):
    tmp = path + TMP_SUFFIX
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)
    if mtime_ns is not None:
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, mtime_ns))


def sanitize_index_file(src: str, dst: str, label: str, show_structure: bool = True) -> dict:
    with open(src, encoding="utf-8") as f:
        index = json.load(f)
    counts: Counter = Counter()
    blank_prompts(index, counts)
    if show_structure:
        report: dict = {}
        string_leaf_report(index, report)
        print(f"[{label}] cache.json string-valued fields after blanking (pattern: total / non-empty):")
        for pattern in sorted(report):
            total, nonempty = report[pattern]
            print(f"    {pattern}: {total} / {nonempty}")
    print(f"[{label}] cache.json prompt fields:")
    for loc, n in sorted(counts.items()):
        print(f"    {loc}: {n}")
    unhandled = [k for k in counts if "UNHANDLED" in k]
    if unhandled:
        raise SystemExit(f"[{label}] unhandled prompt value types: {unhandled}")
    atomic_write_bytes(dst, dump_index(index))
    return {"blanked": sum(n for k, n in counts.items() if "non-empty blanked" in k)}


# ---------------------------------------------------------------------------
# .pt files
# ---------------------------------------------------------------------------

_WORKER: dict = {}


def _init_worker(fill: int, scrub_names: tuple[str, ...]):
    import torch

    torch.set_num_threads(1)
    _WORKER["fill"] = fill
    _WORKER["scrub"] = scrub_names


def _pt_strings(data: dict) -> list[str]:
    """Key paths of string values in a .pt payload that are not known-safe."""
    found = []
    for key, value in data.items():
        if isinstance(value, str) and key not in KNOWN_PT_STRING_KEYS:
            found.append(key)
        elif key == "__sourceless_values" and isinstance(value, dict):
            for sub, sv in value.items():
                if sub == "concept":
                    continue
                if isinstance(sv, str) and sv:
                    found.append(f"{key}.{sub}")
        elif isinstance(value, dict) and key != "__sourceless_values":
            found.append(f"{key}(dict)")
        elif isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value):
            found.append(f"{key}(list)")
    return found


def scrub_payload(data: dict, fill: int, scrub_names) -> tuple[bool, list[str]]:
    """Scrub in place. Returns (changed, scrubbed key names)."""
    import torch

    changed = False
    scrubbed = []
    for name in scrub_names:
        value = data.get(name)
        if torch.is_tensor(value):
            scrubbed.append(name)
            if value.numel() and not bool((value == fill).all()):
                data[name] = torch.full_like(value, fill)
                changed = True
    sv = data.get("__sourceless_values")
    if isinstance(sv, dict):
        for key in PROMPT_KEYS:
            if isinstance(sv.get(key), str) and sv[key]:
                sv[key] = ""
                changed = True
    return changed, scrubbed


def _signature(data: dict) -> dict:
    import torch

    sig = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            sig[key] = (tuple(value.shape), str(value.dtype))
        else:
            sig[key] = type(value).__name__
    return sig


def _compare(src: dict, out: dict, fill: int, scrub_names) -> list[str]:
    """Problems (key names only) between an original payload and its scrubbed form."""
    import torch

    problems = []
    if set(src) != set(out):
        problems.append("key set differs")
        return problems
    for key, sv in src.items():
        ov = out[key]
        if torch.is_tensor(sv):
            if not torch.is_tensor(ov) or sv.shape != ov.shape or sv.dtype != ov.dtype:
                problems.append(f"{key}: shape/dtype")
            elif key in scrub_names:
                if ov.numel() and not bool((ov == fill).all()):
                    problems.append(f"{key}: not all fill")
            elif not torch.equal(sv, ov):
                problems.append(f"{key}: not bit-identical")
        elif key == "__sourceless_values":
            if not isinstance(ov, dict) or set(sv) != set(ov):
                problems.append(f"{key}: keys differ")
            else:
                for sub in sv:
                    if sub in PROMPT_KEYS:
                        if ov[sub] != "":
                            problems.append(f"{key}.{sub}: not blank")
                    elif sv[sub] != ov[sub]:
                        problems.append(f"{key}.{sub}: differs")
        elif sv != ov:
            problems.append(f"{key}: differs")
    return problems


def _process_pt(task: tuple[str, str, bool]) -> dict:
    """Rewrite one .pt (src -> dst; dst == src for in-place) and verify it."""
    import torch

    src, dst, in_place = task
    fill, scrub_names = _WORKER["fill"], _WORKER["scrub"]
    result = {"name": os.path.basename(src), "status": "ok", "problems": [], "strings": [], "sig": None}
    try:
        st = os.stat(src)
        if not in_place:
            with contextlib.suppress(OSError):
                if os.stat(dst).st_mtime_ns == st.st_mtime_ns:
                    result["status"] = "skipped"
                    return result
        data = torch.load(src, weights_only=False, map_location="cpu")
        if not isinstance(data, dict):
            result["status"] = "failed"
            result["problems"].append(f"payload is {type(data).__name__}")
            return result
        result["sig"] = sorted(_signature(data).items())
        original = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        if isinstance(data.get("__sourceless_values"), dict):
            original["__sourceless_values"] = dict(data["__sourceless_values"])
        changed, _ = scrub_payload(data, fill, scrub_names)
        result["strings"] = _pt_strings(data)
        if in_place and not changed:
            result["status"] = "already-clean"
            return result
        tmp = dst + TMP_SUFFIX
        torch.save(data, tmp)
        check = torch.load(tmp, weights_only=False, map_location="cpu")
        problems = _compare(original, check, fill, scrub_names)
        if problems:
            os.remove(tmp)
            result["status"] = "failed"
            result["problems"] = problems
            return result
        os.replace(tmp, dst)
        os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
        result["status"] = "rewritten"
    except Exception as e:  # noqa: BLE001 - reported per file, the run continues
        result["status"] = "failed"
        result["problems"].append(f"{type(e).__name__}")
    return result


def _verify_pair(task: tuple[str, str]) -> dict:
    import torch

    src, dst = task
    fill, scrub_names = _WORKER["fill"], _WORKER["scrub"]
    result = {"name": os.path.basename(src), "problems": [], "strings": []}
    try:
        a = torch.load(src, weights_only=False, map_location="cpu")
        b = torch.load(dst, weights_only=False, map_location="cpu")
        result["problems"] = _compare(a, b, fill, scrub_names)
        result["strings"] = _pt_strings(b)
    except Exception as e:  # noqa: BLE001
        result["problems"].append(type(e).__name__)
    return result


def _verify_clean(path: str) -> dict:
    """In-place verification: the file itself must already be scrubbed."""
    import torch

    fill, scrub_names = _WORKER["fill"], _WORKER["scrub"]
    result = {"name": os.path.basename(path), "problems": [], "strings": []}
    try:
        data = torch.load(path, weights_only=False, map_location="cpu")
        changed, _ = scrub_payload(data, fill, scrub_names)
        if changed:
            result["problems"].append("not scrubbed")
        result["strings"] = _pt_strings(data)
    except Exception as e:  # noqa: BLE001
        result["problems"].append(type(e).__name__)
    return result


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def list_dir(path: str) -> dict[str, os.stat_result]:
    with os.scandir(path) as it:
        return {e.name: e.stat() for e in it if e.is_file(follow_symlinks=False)}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(names) -> dict[str, list[str]]:
    groups = {"pt": [], "index": [], "content_index": [], "tmp": [], "bak": [], "other": []}
    for name in sorted(names):
        if name.endswith(".tmp"):
            groups["tmp"].append(name)
        elif name.endswith(".bak"):
            groups["bak"].append(name)
        elif name.endswith(".pt"):
            groups["pt"].append(name)
        elif name == "cache.json":
            groups["index"].append(name)
        elif name == "content_index.json":
            groups["content_index"].append(name)
        else:
            groups["other"].append(name)
    return groups


def check_content_index(path: str, pt_names: set[str], label: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries") if isinstance(data, dict) else None
    top = sorted(data) if isinstance(data, dict) else type(data).__name__
    hex_keys = sum(1 for k in (entries or {}) if re.fullmatch(r"[0-9a-f]{16}", k))
    pt_values = sum(1 for v in (entries or {}).values() if isinstance(v, str) and v in pt_names)
    n = len(entries or {})
    print(
        f"[{label}] content_index.json: top-level keys {top}; {n} entries; "
        f"{hex_keys} keys are 16-hex content hashes; {pt_values} values are .pt names present in the dir"
    )
    if not isinstance(entries, dict) or hex_keys != n or not all(isinstance(v, str) for v in entries.values()):
        raise SystemExit(f"[{label}] content_index.json has an unexpected structure; not copying it")


def run_pool(fn, tasks, workers: int, fill: int, scrub: tuple[str, ...], label: str, what: str):
    results = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(fill, scrub)
    ) as pool:
        for i, res in enumerate(pool.map(fn, tasks, chunksize=16), 1):
            results.append(res)
            if i % 2000 == 0 or i == len(tasks):
                print(f"[{label}] {what}: {i}/{len(tasks)} ({time.monotonic() - started:.0f}s)", flush=True)
    return results


def summarize(results: list[dict], label: str, what: str):
    status = Counter(r.get("status", "verified") for r in results)
    problems = Counter(p for r in results for p in r["problems"])
    strings = Counter(s for r in results for s in r["strings"])
    sigs = Counter(tuple(k for k, _ in r["sig"]) for r in results if r.get("sig"))
    print(f"[{label}] {what}: statuses {dict(status)}")
    if sigs:
        print(f"[{label}] {what}: .pt key sets seen:")
        for keys, n in sigs.most_common():
            print(f"    {n} x {list(keys)}")
    print(f"[{label}] {what}: files with problems: {sum(1 for r in results if r['problems'])}")
    for p, n in problems.most_common():
        print(f"    {p}: {n}")
    if strings:
        print(f"[{label}] {what}: unexpected string-bearing .pt keys (names only):")
        for s, n in strings.most_common():
            print(f"    {s}: {n}")
    else:
        print(f"[{label}] {what}: no unexpected string-bearing .pt keys")
    return problems


def process_cache(src: str, dst: str | None, args) -> bool:
    in_place = dst is None
    out_dir = src if in_place else dst
    label = os.path.join(os.path.basename(os.path.dirname(src)), os.path.basename(src))
    ok = True
    t0 = time.monotonic()

    src_files = list_dir(src)
    groups = classify(src_files)
    print(
        f"[{label}] source: {len(groups['pt'])} .pt, {len(groups['tmp'])} .tmp (skipped), "
        f"{len(groups['bak'])} .bak, other files {groups['other']}"
    )
    if not groups["index"]:
        raise SystemExit(f"[{label}] no cache.json in {src}")
    if groups["other"]:
        raise SystemExit(f"[{label}] unexpected files {groups['other']}; inspect before running")

    # Snapshot of the originals (staging mode): size+mtime of every file, sha256
    # of the index files and a .pt sample.
    rng = random.Random(1234)
    sample_pts = rng.sample(groups["pt"], min(args.hash_sample, len(groups["pt"])))
    hashed = groups["index"] + groups["content_index"] + sample_pts
    before_stat = {n: (s.st_size, s.st_mtime_ns) for n, s in src_files.items()}
    before_hash = {n: sha256(os.path.join(src, n)) for n in hashed} if not in_place else {}

    if not args.verify_only:
        if not in_place:
            os.makedirs(dst, exist_ok=True)
            need = sum(src_files[n].st_size for n in groups["pt"] + groups["index"] + groups["content_index"])
            have = sum(s.st_size for n, s in list_dir(dst).items() if n in src_files)
            free = shutil.disk_usage(dst).free
            print(f"[{label}] needs ~{(need - have) / 2**30:.1f} GiB more, {free / 2**30:.1f} GiB free")
            if need - have > free - 5 * 2**30:
                raise SystemExit(f"[{label}] not enough free space")
        else:
            with open(os.path.join(src, "cache.json"), encoding="utf-8") as f:
                holds_prompts = count_nonempty_prompts(json.load(f)) > 0
            if holds_prompts:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup = os.path.join(src, f"cache.json.pre-scrub-{stamp}.bak")
                shutil.copy2(os.path.join(src, "cache.json"), backup)
                print(f"[{label}] backed up cache.json -> {os.path.basename(backup)}")
            else:
                print(f"[{label}] cache.json already holds no prompts; no backup taken")

        if groups["content_index"]:
            check_content_index(os.path.join(src, "content_index.json"), set(groups["pt"]), label)
        info = sanitize_index_file(os.path.join(src, "cache.json"), os.path.join(out_dir, "cache.json"), label)
        print(f"[{label}] cache.json: {info['blanked']} non-empty prompt strings blanked")
        if not in_place and groups["content_index"]:
            shutil.copy2(os.path.join(src, "content_index.json"), os.path.join(dst, "content_index.json"))

        tasks = [(os.path.join(src, n), os.path.join(out_dir, n), in_place) for n in groups["pt"]]
        t1 = time.monotonic()
        results = run_pool(_process_pt, tasks, args.workers, args.fill_value, args.scrub, label, "rewrite")
        print(f"[{label}] rewrite took {time.monotonic() - t1:.0f}s")
        if summarize(results, label, "rewrite"):
            ok = False

        if not in_place:
            # Drop stale outputs from earlier runs (only files this script writes).
            extra = [n for n in list_dir(dst) if (n.endswith(".pt") and n not in src_files) or n.endswith(TMP_SUFFIX)]
            for n in extra:
                os.remove(os.path.join(dst, n))
            print(f"[{label}] removed {len(extra)} stale staging files")

    # ---------------- verification ----------------
    t2 = time.monotonic()
    out_files = list_dir(out_dir)
    out_groups = classify(out_files)
    print(f"[{label}] verify: .pt in {len(groups['pt'])}, .pt out {len(out_groups['pt'])}")
    if set(out_groups["pt"]) != set(groups["pt"]):
        print(f"[{label}] verify: .pt NAME SETS DIFFER")
        ok = False
    if not in_place and (out_groups["tmp"] or out_groups["bak"] or out_groups["other"]):
        print(f"[{label}] verify: staging dir has extra non-.pt files: {out_groups['tmp'] + out_groups['bak']}")
        ok = False

    with open(os.path.join(out_dir, "cache.json"), encoding="utf-8") as f:
        out_index = json.load(f)
    remaining = count_nonempty_prompts(out_index)
    print(f"[{label}] verify: cache.json non-empty prompt strings remaining: {remaining}")
    ok &= remaining == 0
    if not in_place:
        with open(os.path.join(src, "cache.json"), encoding="utf-8") as f:
            src_index = json.load(f)
        same_entries = set(src_index.get("entries", {})) == set(out_index.get("entries", {}))
        print(
            f"[{label}] verify: cache.json entries in {len(src_index.get('entries', {}))}, "
            f"out {len(out_index.get('entries', {}))}, same key set: {same_entries}, "
            f"same top-level keys: {sorted(src_index) == sorted(out_index)}"
        )
        blank_prompts(src_index, Counter())
        identical = src_index == out_index
        print(f"[{label}] verify: cache.json == blank_prompts(source cache.json): {identical}")
        ok &= same_entries and identical
        if groups["content_index"]:
            same_ci = sha256(os.path.join(src, "content_index.json")) == sha256(os.path.join(dst, "content_index.json"))
            print(f"[{label}] verify: content_index.json byte-identical copy: {same_ci}")
            ok &= same_ci
        del src_index
        tasks = [(os.path.join(src, n), os.path.join(dst, n)) for n in groups["pt"]]
        vres = run_pool(_verify_pair, tasks, args.workers, args.fill_value, args.scrub, label, "verify")
    else:
        tasks = [os.path.join(src, n) for n in groups["pt"]]
        vres = run_pool(_verify_clean, tasks, args.workers, args.fill_value, args.scrub, label, "verify")
    if summarize(vres, label, "verify"):
        ok = False
    print(f"[{label}] verify took {time.monotonic() - t2:.0f}s")

    if not in_place:
        after = list_dir(src)
        changed = [n for n, s in before_stat.items() if n in after and (after[n].st_size, after[n].st_mtime_ns) != s]
        missing = [n for n in before_stat if n not in after]
        rehashed = sum(1 for n, h in before_hash.items() if sha256(os.path.join(src, n)) != h)
        print(
            f"[{label}] originals: {len(before_stat)} files snapshotted, {len(changed)} size/mtime changed, "
            f"{len(missing)} missing, {len(before_hash)} sha256-checked ({rehashed} differ)"
        )
        ok &= not changed and not missing and rehashed == 0

    if args.functional_check:
        ok &= functional_check(src, out_dir, args, label)

    if in_place:
        # Backups beside the index are untouched copies of the old index (or
        # its .pt-free predecessors) and still hold the captions.
        baks = sorted(n for n in list_dir(out_dir) if n.endswith(".bak"))
        if baks:
            print(f"[{label}] NOTE: {len(baks)} .bak file(s) here still hold caption text: {baks}")

    print(f"[{label}] {'OK' if ok else 'PROBLEMS FOUND'} ({time.monotonic() - t0:.0f}s total)")
    return ok


# ---------------------------------------------------------------------------
# functional check: read-only sourceless image + text pipeline
# ---------------------------------------------------------------------------


def functional_check(src_text: str, text_dir: str, args, label: str) -> bool:
    from mgds.MGDS import MGDS
    from mgds.OutputPipelineModule import OutputPipelineModule
    from mgds.PipelineModule import PipelineState
    from mgds.pipelineModules.PlaceholderModule import PlaceholderModule
    from mgds.pipelineModules.SmartDiskCache import SmartDiskCache

    import torch

    class ReadOnlySmartDiskCache(SmartDiskCache):
        # The stock loader deletes cache.json.tmp/.bak beside a good index;
        # never touch the user's dirs from a check.
        def _load_cache_index(self):
            with open(self._get_cache_json_path(), encoding="utf-8") as f:
                data = json.load(f)
            self._migrate_loaded_index(data)
            return data

        def _save_cache_index(self, *a, **k):
            raise AssertionError("functional check tried to write cache.json")

        def _save_content_index(self):
            pass

        def _ensure_blank_sentinel(self):
            pass

    image_dir = os.path.join(os.path.dirname(src_text), "image")
    with open(os.path.join(text_dir, "cache.json"), encoding="utf-8") as f:
        modeltype = next(iter(json.load(f)["entries"].values()))["modeltype"]
    watched = [os.path.join(text_dir, "cache.json"), os.path.join(image_dir, "cache.json")]
    before = {p: (os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in watched}
    image_before = {n: (s.st_size, s.st_mtime_ns) for n, s in list_dir(image_dir).items()}

    def build(cache_dir: str, paired: bool):
        text_cache = ReadOnlySmartDiskCache(
            cache_dir=cache_dir,
            split_names=["tokens", "tokens_mask", "text_encoder_hidden_state"],
            aggregate_names=[],
            modeltype=modeltype,
            source_path_in_name="sample_prompt_path",
            sourceless=True,
            content_key_in_name="prompt",
        )
        modules = [PlaceholderModule()]
        if paired:
            # Image cache as the sourceless anchor, exactly like training:
            # text rows are resolved through its linked_paths.
            modules.append(
                ReadOnlySmartDiskCache(
                    cache_dir=image_dir,
                    split_names=["latent_image"],
                    aggregate_names=[],
                    modeltype=modeltype,
                    source_path_in_name="image_path",
                    sourceless=True,
                )
            )
        modules.append(text_cache)
        ds = MGDS(
            device=torch.device("cpu"),
            concepts=[{"name": "functional-check", "path": "functional-check"}],
            settings={},
            definition=[
                modules,
                OutputPipelineModule(names=["tokens", "tokens_mask", "text_encoder_hidden_state"]),
            ],
            batch_size=1,
            state=PipelineState(),
            seed=42,
        )
        ds.start_next_epoch()
        return text_cache

    def missing_count(err: Exception) -> str:
        m = re.search(r"(\d+) required", str(err))
        return m.group(1) if m else type(err).__name__

    t0 = time.monotonic()
    try:
        text_cache = build(text_dir, paired=True)
        mode = "image-anchored"
    except RuntimeError as err:
        # Training filters anchor rows (e.g. adaptive-resolution eligibility)
        # with config this check doesn't have; an unfiltered anchor can name
        # image rows whose caption entry was never built. Compare against the
        # original text cache to show the gap predates sanitizing.
        try:
            build(src_text, paired=True)
            original = "original cache aligns"
        except RuntimeError as err_original:
            original = f"original cache: {missing_count(err_original)} missing too"
        print(
            f"[{label}] functional: image-anchored alignment (unfiltered anchor rows): "
            f"{missing_count(err)} text entries missing; {original}"
        )
        text_cache = build(text_dir, paired=False)
        mode = "text-only"
    n_rows = text_cache.length()
    print(
        f"[{label}] functional: sourceless {mode} init OK in {time.monotonic() - t0:.0f}s; "
        f"{len(text_cache._sourceless_filepaths)} text rows, {n_rows} output samples"
    )
    get_input_index = text_cache._SmartDiskCache__get_input_index
    rng = random.Random(7)
    picks = rng.sample(range(n_rows), min(args.functional_check, n_rows))
    stats = Counter()
    for out_index in picks:
        item = text_cache.get_item(out_index, "text_encoder_hidden_state")
        group_key, in_variation, _, in_index = get_input_index(text_cache.current_variation, out_index)
        entry = text_cache.cache_index["entries"][text_cache._sourceless_filepaths[in_index]]
        variation = in_variation % text_cache.group_variations[group_key]
        cache_file = text_cache._sourceless_variant(entry, in_index)["cache_file"]
        original = torch.load(
            os.path.join(src_text, f"{cache_file}_{variation + 1}.pt"), weights_only=False, map_location="cpu"
        )
        stats["loaded"] += 1
        stats["hidden_state_equal_original"] += bool(
            torch.equal(item["text_encoder_hidden_state"], original["text_encoder_hidden_state"])
        )
        stats["tokens_mask_equal_original"] += bool(torch.equal(item["tokens_mask"], original["tokens_mask"]))
        stats["tokens_all_fill"] += bool((item["tokens"] == args.fill_value).all())
        stats["tokens_shape_equal_original"] += item["tokens"].shape == original["tokens"].shape
        prompts = [item.get(k) for k in PROMPT_KEYS if k in item]
        stats["rows_with_prompt_key"] += bool(prompts)
        stats["prompt_values_blank"] += all(p == "" for p in prompts) and bool(prompts)
        stats["concept_is_dict"] += isinstance(item.get("concept"), dict)
    print(f"[{label}] functional: {dict(stats)}")
    after = {p: (os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in watched}
    image_after = {n: (s.st_size, s.st_mtime_ns) for n, s in list_dir(image_dir).items()}
    untouched = before == after and image_before == image_after
    print(f"[{label}] functional: cache.json files and image dir untouched by the check: {untouched}")
    n = stats["loaded"]
    return (
        untouched
        and n > 0
        and stats["hidden_state_equal_original"] == n
        and stats["tokens_all_fill"] == n
        and stats["prompt_values_blank"] == stats["rows_with_prompt_key"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="text cache dirs (the folder holding cache.json)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staging-root", help="write sanitized copies to <root>/<parent>/<name>")
    mode.add_argument("--in-place", action="store_true", help="scrub the caches themselves")
    parser.add_argument("--confirm-in-place", action="store_true", help="required together with --in-place")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--fill", choices=["zero", "pad"], default="zero", help="token fill value (default zero)")
    parser.add_argument("--tokenizer-dir", help="tokenizer folder for --fill pad (default: Z-Image in HF cache)")
    parser.add_argument("--scrub", default=",".join(DEFAULT_SCRUB_NAMES), help="comma-separated token tensor names")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--functional-check", type=int, default=0, metavar="N")
    parser.add_argument("--hash-sample", type=int, default=64, help="source .pt files to sha256 before/after")
    args = parser.parse_args()

    if args.in_place and not args.confirm_in_place and not args.verify_only:
        parser.error("--in-place rewrites the caches; pass --confirm-in-place too")
    args.scrub = tuple(s for s in args.scrub.split(",") if s)
    if args.fill == "pad":
        args.fill_value, where = find_pad_id(args.tokenizer_dir)
        print(f"fill: pad id {args.fill_value} (pad_token from {where})")
    else:
        args.fill_value = 0
        print("fill: 0 (matches mgds scrub_split_names)")
    print(f"scrubbing tensors {list(args.scrub)} + prompt strings {list(PROMPT_KEYS)}")

    all_ok = True
    for src in args.sources:
        src = os.path.normpath(src)
        dst = None
        if args.staging_root:
            dst = os.path.join(args.staging_root, os.path.basename(os.path.dirname(src)), os.path.basename(src))
            if os.path.normcase(os.path.abspath(dst)) == os.path.normcase(os.path.abspath(src)):
                parser.error("staging dir resolves to the source dir")
        all_ok &= process_cache(src, dst, args)
    print("ALL OK" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
