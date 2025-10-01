#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

def convert_safetensors_file(src_path: Path, dst_path: Path, out_dtype: torch.dtype):
    # Read metadata (without loading all tensors)
    meta = {}
    try:
        with safe_open(str(src_path), framework="pt", device="cpu") as f:
            meta = dict(f.metadata()) if f.metadata() is not None else {}
    except Exception as e:
        print(f"[WARN] Could not read metadata for {src_path.name}: {e}")

    # Load all tensors (dict[str, Tensor]) on CPU
    tensors = load_file(str(src_path), device="cpu")

    # Cast only floating tensors to desired dtype
    out = {}
    for k, t in tensors.items():
        if t.is_floating_point():
            # Skip if already desired dtype
            out[k] = t.to(dtype=out_dtype, copy=True)
        else:
            out[k] = t  # keep ints/bools/etc as-is

    # Update metadata with a breadcrumb
    meta = dict(meta)
    meta["converted_by"] = "bf16<->fp16 converter"
    meta["converted_target_dtype"] = str(out_dtype).replace("torch.", "")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dst_path), metadata=meta)
    print(f"[OK] {src_path.name}  ->  {dst_path.name}")

def is_index_json(p: Path) -> bool:
    if p.suffix != ".json":
        return False
    try:
        j = json.loads(p.read_text())
        return "weight_map" in j and "metadata" in j
    except Exception:
        return False

def convert_folder_or_file(src: Path, dst: Path, target: str):
    if target not in DTYPE_MAP:
        raise ValueError("target must be one of: bf16, fp16")
    out_dtype = DTYPE_MAP[target]

    if src.is_file():
        if src.suffix == ".safetensors":
            if dst.is_dir():
                dst = dst / src.name  # put result in the directory
            convert_safetensors_file(src, dst, out_dtype)
        elif is_index_json(src):
            # Convert shards listed in an index.json (write new index next to output shards)
            src_idx = json.loads(src.read_text())
            weight_map = src_idx.get("weight_map", {})
            # Determine output directory
            out_dir = dst if dst.is_dir() or dst.suffix != ".safetensors" else dst.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            new_weight_map = {}
            for tensor_name, shard_rel in weight_map.items():
                shard_src = src.parent / shard_rel
                shard_dst = out_dir / shard_rel  # same filenames
                convert_safetensors_file(shard_src, shard_dst, out_dtype)
                new_weight_map[tensor_name] = shard_rel

            # Copy/adjust index metadata
            new_index = {
                "metadata": dict(src_idx.get("metadata", {})),
                "weight_map": new_weight_map,
                "format": src_idx.get("format", "pt"),
            }
            new_index["metadata"]["converted_by"] = "bf16<->fp16 converter"
            new_index["metadata"]["converted_target_dtype"] = target
            (out_dir / src.name).write_text(json.dumps(new_index, indent=2))
            print(f"[OK] Wrote updated index: {(out_dir / src.name)}")
        else:
            raise ValueError("Unrecognized file. Provide a .safetensors file or an index.json.")
    else:
        # Directory: convert every .safetensors in it, and copy/refresh index.json if present
        dst.mkdir(parents=True, exist_ok=True)
        idx_path = None
        for p in src.iterdir():
            if p.suffix == ".safetensors":
                convert_safetensors_file(p, dst / p.name, out_dtype)
            elif p.name == "model.safetensors.index.json" or (p.suffix == ".json" and is_index_json(p)):
                idx_path = p

        if idx_path is not None:
            # Re-use the helper by pointing it at the index.json so it rewrites it in dst
            convert_folder_or_file(idx_path, dst, target)

def main():
    ap = argparse.ArgumentParser(description="Convert between bf16 and fp16 safetensors.")
    ap.add_argument("--in", dest="inp", required=True, help="Input .safetensors file, index.json, or folder.")
    ap.add_argument("--out", dest="outp", required=True, help="Output file or folder.")
    ap.add_argument("--to", dest="target", required=True, choices=["bf16", "fp16"], help="Target dtype.")
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.outp)
    convert_folder_or_file(src, dst, args.target)

if __name__ == "__main__":
    # Quick sanity check for CUDA users: conversion runs on CPU by default.
    # If you need GPU for speed, you can move tensors .to("cuda").to(dtype).to("cpu") before saving.
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)