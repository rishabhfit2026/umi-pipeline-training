"""
Run to check LinearNormalizer and data pipeline:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    python check_rdt2_more.py
"""
import os, sys
sys.path.insert(0, "/home/rishabh/Downloads/umi-pipeline-training/RDT2")
RDT2_DIR = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"

# Check LinearNormalizer
print("="*60)
print("models/normalizer/ contents:")
norm_dir = f"{RDT2_DIR}/models/normalizer"
for f in os.listdir(norm_dir):
    print(f"  {f}")

# Read normalizer __init__ or main file
for fname in os.listdir(norm_dir):
    if fname.endswith(".py"):
        print(f"\n--- {fname} ---")
        with open(f"{norm_dir}/{fname}") as f:
            print(f.read()[:800])

# Check vla_trainer.py for batch_predict_action
print("\n\n" + "="*60)
print("vla_trainer.py (first 80 lines):")
with open(f"{RDT2_DIR}/vla_trainer.py") as f:
    lines = f.readlines()
for i, l in enumerate(lines[:80]):
    print(f"{i+1:>3}: {l}", end="")

# Check utils.py batch_predict_action
print("\n\n" + "="*60)
print("utils.py full:")
with open(f"{RDT2_DIR}/utils.py") as f:
    print(f.read()[:3000])

# Check configs post_train.yaml full
print("\n\n" + "="*60)
print("configs/rdt/post_train.yaml full:")
with open(f"{RDT2_DIR}/configs/rdt/post_train.yaml") as f:
    print(f.read())