#!/usr/bin/env python3
import argparse
from safetensors import safe_open
import torch

def main():
    parser = argparse.ArgumentParser(description="Print tensor dtype counts in a safetensors file.")
    parser.add_argument("path", help="Path to the .safetensors file")
    args = parser.parse_args()
    
    with safe_open(args.path, framework="pt", device="cpu") as f:
        dtypes = {}
        for k in f.keys():
            t = f.get_tensor(k)
            dtypes.setdefault(str(t.dtype), 0)
            dtypes[str(t.dtype)] += 1
    print(dtypes)

if __name__ == "__main__":
    main()
