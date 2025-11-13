# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, argparse, json, time
# from typing import Optional, Dict, Any, List
# import urllib.request, urllib.error

# from datasets import load_dataset, load_from_disk, Dataset, disable_caching

# OUT = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/pixmo_cap"
# PROMPT = "Describe this image."

# # ---------------- basics ----------------
# def get_rank_worldsize() -> tuple[int, int]:
#     if "SLURM_ARRAY_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_MAX" in os.environ:
#         r = int(os.environ["SLURM_ARRAY_TASK_ID"]); w = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1; return r, w
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
#     if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
#         return int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
#     return 0, 1

# def fetch_bytes(url: str, timeout: float = 6.0, retries: int = 2, backoff: float = 0.6) -> Optional[bytes]:
#     for i in range(retries):
#         try:
#             with urllib.request.urlopen(url, timeout=timeout) as r:
#                 return r.read()
#         except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
#             if i + 1 == retries: return None
#             time.sleep(backoff * (2 ** i)) # exponential backoff just in case
#         except Exception:
#             return None
#     return None

# def to_examples_batch(batch: Dict[str, List[Any]], user_prompt: str) -> Dict[str, Any]:
#     urls = batch.get("image_url") or []
#     caps = batch.get("caption") or []
#     out_convs, out_mods = [], []
#     for url, cap in zip(urls, caps):
#         u = (url or "").strip(); c = (cap or "").strip()
#         if not u or not c:
#             out_convs.append(None); out_mods.append(None); continue
#         b = fetch_bytes(u)
#         if not b:
#             out_convs.append(None); out_mods.append(None); continue
#         out_convs.append([
#             {"role": "user", "content": f"<|reserved_special_token_0|>{user_prompt}"},
#             {"role": "assistant", "content": c},
#         ])
#         out_mods.append([{"type": "image", "value": {"bytes": b, "path": None}}])
#     return {"conversations": out_convs, "modalities": out_mods}

# def process(shard: Dataset, user_prompt: str, num_proc: Optional[int],
#             batch_size: int, writer_batch_size: int) -> Dataset:
#     disable_caching()  # reduce RAM spikes and cache noise
#     mapped = shard.map(
#         lambda b: to_examples_batch(b, user_prompt),
#         batched=True,
#         batch_size=batch_size,
#         remove_columns=shard.column_names,
#         num_proc=num_proc,
#         writer_batch_size=writer_batch_size,
#         load_from_cache_file=False,
#         desc=f"Transforming (batched={batch_size}, writer_batch={writer_batch_size})",
#     )
#     cleaned = mapped.filter(
#         lambda c, m: c is not None and m is not None,
#         input_columns=["conversations", "modalities"],
#     )
#     return cleaned

# # ---------------- main ----------------
# def main():
#     ap = argparse.ArgumentParser(description="pixmo-cap → Arrow shards with raw image bytes (no c10d, no global sync).")
#     ap.add_argument("--output_dir", default=OUT)
#     ap.add_argument("--prompt", default=PROMPT)
#     ap.add_argument("--num_proc", type=int, default=None)
#     ap.add_argument("--batch_size", type=int, default=int(os.environ.get("PIXMO_BATCH_SIZE", 16)))
#     ap.add_argument("--writer_batch_size", type=int, default=int(os.environ.get("PIXMO_WRITER_BATCH_SIZE", 64)))
#     ap.add_argument("--shards_subdir", default="shards")
#     args = ap.parse_args()

#     rank, world_size = get_rank_worldsize()
#     shards_root = os.path.join(args.output_dir, args.shards_subdir)
#     if rank == 0:
#         os.makedirs(args.output_dir, exist_ok=True)
#         os.makedirs(shards_root, exist_ok=True)

#     raw = load_dataset("allenai/pixmo-cap", split="train")
#     shard = raw.shard(num_shards=world_size, index=rank, contiguous=True)

#     if args.num_proc is None:
#         cpu = os.cpu_count() or 8
#         # keep per-rank pressure moderate; too many procs => OOM
#         args.num_proc = max(4, min(8, cpu // max(1, world_size)))

#     out = process(
#         shard,
#         args.prompt,
#         num_proc=args.num_proc,
#         batch_size=max(4, args.batch_size),
#         writer_batch_size=max(16, args.writer_batch_size),
#     )

#     rank_out = os.path.join(shards_root, f"rank_{rank}")
#     os.makedirs(rank_out, exist_ok=True)
#     out.save_to_disk(rank_out)

#     rows = len(out)
#     with open(os.path.join(rank_out, "meta.json"), "w") as f:
#         json.dump({"rank": rank, "world_size": world_size, "rows": rows}, f)

#     print(f"✅ Saved shard (rank {rank}) with {rows} rows under {rank_out}")

#     # Optional opportunistic merge (non-blocking): set PIXMO_AUTO_MERGE=1
#     if os.environ.get("PIXMO_AUTO_MERGE", "0") == "1" and rank == 0:
#         parts = []
#         for d in sorted(os.listdir(shards_root)):
#             p = os.path.join(shards_root, d)
#             if os.path.isdir(p) and d.startswith("rank_"):
#                 try:
#                     parts.append(load_from_disk(p))
#                 except Exception:
#                     pass
#         if parts:
#             from datasets import concatenate_datasets
#             final = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
#             final.save_to_disk(args.output_dir)
#             print(f"✅ Saved Dataset with {len(final)} rows to {args.output_dir}")

#     print(f"[rank {rank}/{world_size}] wrote {rows} rows to {rank_out}")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import List

from datasets import load_from_disk, concatenate_datasets, Dataset

DEF_OUT = "/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/pixmo_cap"
TMP_SUBDIR = "_tmp_pixmo"

TARGET_COLS = ["conversations", "modalities"]

def collect_rank_paths(tmp_root: str) -> List[str]:
    if not os.path.isdir(tmp_root):
        return []
    dirs = []
    for name in os.listdir(tmp_root):
        if not name.startswith("rank_"):
            continue
        try:
            _ = int(name.split("_", 1)[1])
        except Exception:
            continue
        dirs.append(name)
    # sort by numeric rank suffix
    dirs.sort(key=lambda n: int(n.split("_", 1)[1]))
    return [os.path.join(tmp_root, d) for d in dirs]

def normalize_cols(ds: Dataset) -> Dataset:
    cols = [c for c in TARGET_COLS if c in ds.column_names]
    if cols != TARGET_COLS:
        # ensure both exist; if missing, create empty columns to allow concat safety
        for need in TARGET_COLS:
            if need not in ds.column_names:
                ds = ds.add_column(need, [None] * len(ds))
        cols = TARGET_COLS
    extra = [c for c in ds.column_names if c not in TARGET_COLS]
    if extra:
        ds = ds.remove_columns(extra)
    return ds

def main():
    ap = argparse.ArgumentParser(description="Merge per-rank pixmo shards into a single HF dataset.")
    ap.add_argument("--output_dir", default=DEF_OUT, help="Final dataset directory (same as preprocess --output_dir).")
    ap.add_argument("--tmp_subdir", default=TMP_SUBDIR, help="Subdir containing rank_* shards.")
    args = ap.parse_args()

    tmp_root = os.path.join(args.output_dir, args.tmp_subdir)
    rank_dirs = collect_rank_paths(tmp_root)
    if not rank_dirs:
        raise SystemExit(f"No rank_* shards found under {tmp_root}")

    parts = []
    for p in rank_dirs:
        try:
            ds = load_from_disk(p)
            ds = normalize_cols(ds)
            parts.append(ds)
        except Exception as e:
            print(f"[warn] skip {p}: {e}")

    if not parts:
        raise SystemExit("No valid shard could be loaded.")

    final = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    # Safety: preserve only target columns
    final = normalize_cols(final)

    # Save to the SAME output_dir (root), replacing any prior incomplete save
    final.save_to_disk(args.output_dir)
    print(f"✅ Saved Dataset with {len(final)} rows to {args.output_dir}")
    print(final)  # should show: features ['conversations','modalities']

if __name__ == "__main__":
    main()
