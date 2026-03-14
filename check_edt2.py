"""
Run to check RDT2 model structure:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python check_rdt2.py
"""
import os, sys
sys.path.insert(0, "/home/rishabh/Downloads/umi-pipeline-training/RDT2")

RDT2_DIR = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"

# 1. Read train.py to understand how model is loaded
print("="*60)
print("train.py (first 100 lines):")
print("="*60)
with open(f"{RDT2_DIR}/train.py") as f:
    lines = f.readlines()
for i, l in enumerate(lines[:100]):
    print(f"{i+1:>3}: {l}", end="")

print("\n\n" + "="*60)
print("main.py (first 80 lines):")
print("="*60)
with open(f"{RDT2_DIR}/main.py") as f:
    lines = f.readlines()
for i, l in enumerate(lines[:80]):
    print(f"{i+1:>3}: {l}", end="")

print("\n\n" + "="*60)
print("models/ directory:")
print("="*60)
for f in os.listdir(f"{RDT2_DIR}/models"):
    print(f"  {f}")

print("\n\n" + "="*60)
print("configs/rdt/ directory:")
print("="*60)
rdt_cfg = f"{RDT2_DIR}/configs/rdt"
if os.path.exists(rdt_cfg):
    for f in os.listdir(rdt_cfg):
        print(f"  {f}")
        if f.endswith(".json") or f.endswith(".yaml"):
            with open(f"{rdt_cfg}/{f}") as fh:
                print(fh.read()[:500])

print("\n\n" + "="*60)
print("utils.py (first 50 lines):")
print("="*60)
with open(f"{RDT2_DIR}/utils.py") as f:
    lines = f.readlines()
for i, l in enumerate(lines[:50]):
    print(f"{i+1:>3}: {l}", end="")