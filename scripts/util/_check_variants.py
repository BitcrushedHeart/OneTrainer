import json

with open(r"F:/workspace/SoReal!/cache/image/cache.json", "r", encoding="utf-8") as f:
    idx = json.load(f)

targets = ("0087647452fd", "73c3034396ce")

for fp, e in idx.get("entries", {}).items():
    variants = e.get("variants", {})
    cfs = " ".join(str(v.get("cache_file") or "") for v in variants.values())
    if any(t in cfs for t in targets):
        print("SOURCE:", fp)
        print("  mtime:", e.get("mtime"), "hash:", e.get("hash"))
        for k, v in variants.items():
            print(f"  variant key={k!r} cache_file={v.get('cache_file')!r}")
        print()
