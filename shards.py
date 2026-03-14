"""
Run this to inspect your .tar shards:
    cd /home/rishabh/Downloads/umi-pipeline-training
    source umi_env/bin/activate
    python check_shards.py
"""
import os, tarfile, io, json
import numpy as np

SHARDS_DIR = "/home/rishabh/Downloads/umi-pipeline-training/shards"

# Find all tar files
tar_files = sorted([
    os.path.join(SHARDS_DIR, f)
    for f in os.listdir(SHARDS_DIR)
    if f.endswith(".tar")
])
print(f"Total .tar files: {len(tar_files)}")

# Inspect first tar file
f = tar_files[0]
print(f"\nInspecting: {f}")

with tarfile.open(f, "r") as tar:
    members = tar.getmembers()
    print(f"Files inside tar ({len(members)} total):")
    for m in members[:20]:
        print(f"  {m.name}  size={m.size}")

    print(f"\n--- Reading each file type ---")
    for m in members:
        ext = os.path.splitext(m.name)[1]
        try:
            data = tar.extractfile(m)
            if data is None:
                continue
            raw = data.read()

            if ext == ".json":
                obj = json.loads(raw)
                print(f"\n{m.name} (JSON):")
                print(f"  keys: {list(obj.keys()) if isinstance(obj, dict) else type(obj)}")
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        print(f"    {k}: {v}")

            elif ext in [".npy", ".npz"]:
                arr = np.load(io.BytesIO(raw), allow_pickle=True)
                if hasattr(arr, 'shape'):
                    print(f"\n{m.name} (numpy): shape={arr.shape}, dtype={arr.dtype}")
                    if arr.ndim <= 2:
                        print(f"  values: {arr[:3]}")
                else:
                    print(f"\n{m.name} (npz): keys={list(arr.keys())}")
                    for k in arr.keys():
                        print(f"  {k}: shape={arr[k].shape}")

            elif ext in [".pt", ".pth"]:
                import torch
                obj = torch.load(io.BytesIO(raw), map_location="cpu")
                print(f"\n{m.name} (torch): type={type(obj)}")
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if hasattr(v, 'shape'):
                            print(f"  {k}: shape={v.shape}")
                elif hasattr(obj, 'shape'):
                    print(f"  shape={obj.shape}")

            elif ext in [".jpg", ".png", ".jpeg"]:
                print(f"\n{m.name} (image): size={len(raw)} bytes")

            elif ext == ".txt":
                print(f"\n{m.name} (text): {raw[:200]}")

            else:
                print(f"\n{m.name}: unknown ext={ext}, size={len(raw)}")

        except Exception as e:
            print(f"\n{m.name}: error={e}")