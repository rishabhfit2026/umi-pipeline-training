"""
Run to see full train.py and data pipeline:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    python check_train_full.py 2>&1 | head -300
"""
import os, sys
sys.path.insert(0, "/home/rishabh/Downloads/umi-pipeline-training/RDT2")
RDT2_DIR = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"

# Full train.py
print("="*60)
print("train.py FULL:")
print("="*60)
with open(f"{RDT2_DIR}/train.py") as f:
    print(f.read())

# data directory
print("\n\n" + "="*60)
print("data/ directory:")
print("="*60)
for root, dirs, files in os.walk(f"{RDT2_DIR}/data"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    level = root.replace(f"{RDT2_DIR}/data", "").count(os.sep)
    indent = "  " * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        if f.endswith(".py"):
            print(f"{indent}  {f}")

# Read data/utils.py
print("\n\n" + "="*60)
print("data/utils.py (first 100 lines):")
print("="*60)
utils_path = f"{RDT2_DIR}/data/utils.py"
if os.path.exists(utils_path):
    with open(utils_path) as f:
        for i, l in enumerate(f.readlines()[:100]):
            print(f"{i+1:>3}: {l}", end="")

# VLATrainer compute_loss
print("\n\n" + "="*60)
print("vla_trainer.py compute_loss section:")
print("="*60)
with open(f"{RDT2_DIR}/vla_trainer.py") as f:
    content = f.read()
    # Find compute_loss
    idx = content.find("compute_loss")
    if idx > 0:
        print(content[idx:idx+2000])
    else:
        idx = content.find("def training_step")
        if idx > 0:
            print(content[idx:idx+2000])
        else:
            print(content[3000:6000])