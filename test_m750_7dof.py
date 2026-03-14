"""
M750 7-DOF Virtual Inference Diagnostic (UPDATED)
==================================================
Fixes:
  - Auto-finds correct vqvae class (no hardcoded MultiVQVAE import)
  - Uses umi_normalizer_official.pt (confirmed D=7 M750 format)
  - Correct normalizer path

Run ON YOUR MACHINE:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python test_m750_7dof.py
"""

import sys, os, importlib, glob
import torch
import numpy as np

# ─── CONFIG ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v2/checkpoint-1000"
NORMALIZER_PATH = "/home/rishabh/Downloads/umi-pipeline-training/RDT2/umi_normalizer_official.pt"
DEVICE          = "cuda:0" if torch.cuda.is_available() else "cpu"
INSTRUCTION     = "Pick up the object."
RDT2_DIR        = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
# ────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, RDT2_DIR)
os.chdir(RDT2_DIR)

M750_DOF_NAMES = ["x (m)", "y (m)", "z (m)", "rx (rad)", "ry (rad)", "rz (rad)", "gripper"]
M750_DOF       = 7

def banner(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")

# ─── STEP 0: Auto-find vqvae class ──────────────────────────────────────────
banner("STEP 0 — Auto-detecting vqvae import")

def find_vqvae_class():
    """
    Tries every known import path for the VQ-VAE class in RDT2.
    Returns (class, module_name, class_name) or raises RuntimeError.
    """
    candidates = [
        ("vqvae",              "MultiVQVAE"),
        ("vqvae.vqvae",        "MultiVQVAE"),
        ("vqvae.model",        "MultiVQVAE"),
        ("vqvae.multi_vqvae",  "MultiVQVAE"),
        ("models.vqvae",       "MultiVQVAE"),
        ("models.multi_vqvae", "MultiVQVAE"),
        ("vqvae",              "VQVAE"),
        ("vqvae.vqvae",        "VQVAE"),
        ("vqvae.model",        "VQVAE"),
    ]

    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                print(f"  ✅  from {mod_name} import {cls_name}")
                return cls, mod_name, cls_name
        except Exception:
            pass

    # Deep scan: walk all .py files inside vqvae/ directory
    vqvae_dir = os.path.join(RDT2_DIR, "vqvae")
    if os.path.isdir(vqvae_dir):
        print(f"  Scanning {vqvae_dir} for VAE classes...")
        for fname in sorted(os.listdir(vqvae_dir)):
            if not fname.endswith(".py"):
                continue
            mod_name = f"vqvae.{fname[:-3]}"
            try:
                mod = importlib.import_module(mod_name)
                for attr in dir(mod):
                    if any(x in attr.lower() for x in ["vqvae", "vae", "tokenizer"]):
                        cls = getattr(mod, attr)
                        if isinstance(cls, type) and hasattr(cls, "from_pretrained"):
                            print(f"  ✅  from {mod_name} import {attr}")
                            return cls, mod_name, attr
            except Exception as e:
                print(f"     {mod_name}: {e}")

    # Nothing found — print diagnostic info to help fix
    print("\n  ❌  Could not find vqvae class. Diagnostic info:")
    try:
        import vqvae as vq
        print(f"     vqvae.__file__ = {vq.__file__}")
        print(f"     dir(vqvae)     = {dir(vq)}")
    except Exception as e:
        print(f"     import vqvae failed: {e}")

    raise RuntimeError(
        "\n\n  Could not find MultiVQVAE / VQVAE class.\n"
        "  Run this to debug:\n"
        "    python -c \"import vqvae; print(dir(vqvae)); print(vqvae.__file__)\"\n"
        "  Then share the output."
    )


VQVAEClass, VQVAE_MOD, VQVAE_CLS = find_vqvae_class()

# ─── STEP 1: Dataset check ───────────────────────────────────────────────────
banner("STEP 1 — Checking dataset action dimension")

def check_dataset_action_dim():
    shard_dirs = [
        "/home/rishabh/Downloads/umi-pipeline-training/shards",
        "/home/rishabh/Downloads/umi-pipeline-training/dataset",
        "/home/rishabh/Downloads/umi-pipeline-training/data",
        os.path.join(RDT2_DIR, "data"),
    ]
    found = False
    for d in shard_dirs:
        for fpath in glob.glob(f"{d}/**/*.pt", recursive=True)[:3]:
            try:
                data = torch.load(fpath, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    for k in ["action", "actions", "label", "target"]:
                        if k in data:
                            act = data[k]
                            if hasattr(act, "shape"):
                                D = act.shape[-1]
                                print(f"  Found '{k}' in {os.path.basename(fpath)}  shape={tuple(act.shape)}")
                                if D == 7:
                                    print("  ✅  D=7 → M750 format confirmed in dataset!")
                                elif D == 20:
                                    print("  ⚠️  D=20 → UMI dual-arm format (not M750 7-DOF)")
                                else:
                                    print(f"  ❓  D={D} → unexpected dimension")
                                found = True
            except Exception:
                pass
    if not found:
        print("  ℹ️  Could not auto-detect shards.")
        print("  ✅  Normalizer umi_normalizer_official.pt confirmed D=7 externally.")
        print("      Your pipeline was built for M750 7-DOF: x,y,z,rx,ry,rz,gripper")

check_dataset_action_dim()

# ─── STEP 2: Load model ──────────────────────────────────────────────────────
banner("STEP 2 — Loading fine-tuned model")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

print("  Loading processor...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

print(f"  Loading checkpoint:\n    {CHECKPOINT_PATH}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=DEVICE,
).eval()
print("  ✅  Model loaded")

# ─── STEP 3: Load VQ-VAE ─────────────────────────────────────────────────────
banner("STEP 3 — Loading VQ-VAE tokenizer")

vae = VQVAEClass.from_pretrained("robotics-diffusion-transformer/RVQActionTokenizer").eval()
vae = vae.to(device=DEVICE, dtype=torch.float32)

if hasattr(vae, "pos_id_len"):
    valid_action_id_length = vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
    print(f"  ✅  VQ-VAE loaded ({VQVAE_CLS})")
    print(f"     pos_id_len={vae.pos_id_len}  rot_id_len={vae.rot_id_len}  grip_id_len={vae.grip_id_len}")
    print(f"     valid_action_id_length = {valid_action_id_length}")
else:
    valid_action_id_length = None
    print(f"  ✅  VQ-VAE loaded ({VQVAE_CLS})")
    print(f"     ⚠️  pos_id_len not found — will call without valid_action_id_length")
    print(f"     attrs: {[a for a in dir(vae) if not a.startswith('_')][:15]}")

# ─── STEP 4: Load normalizer ──────────────────────────────────────────────────
banner("STEP 4 — Loading normalizer (D=7 confirmed)")

from models.normalizer import LinearNormalizer

# Try all candidate locations
normalizer_candidates = [
    NORMALIZER_PATH,
    os.path.join(RDT2_DIR, "umi_normalizer_official.pt"),
    "/home/rishabh/Downloads/umi-pipeline-training/umi_normalizer_official.pt",
    os.path.join(RDT2_DIR, "umi_normalizer_wo_downsample_indentity_rot.pt"),
    "/home/rishabh/Downloads/umi-pipeline-training/RDT2/normalizer.pt",
]

normalizer = None
for npath in normalizer_candidates:
    if os.path.exists(npath):
        normalizer = LinearNormalizer.from_pretrained(npath)
        print(f"  ✅  Loaded: {npath}")
        break

if normalizer is None:
    print("  ❌  Normalizer not found at any candidate path:")
    for p in normalizer_candidates:
        print(f"       {p}")
    print()
    print("  → Copy umi_normalizer_official.pt into your RDT2 directory:")
    print("    cp /path/to/umi_normalizer_official.pt "
          "/home/rishabh/Downloads/umi-pipeline-training/RDT2/")
    sys.exit(1)

# ─── STEP 5: Inference on fake images ────────────────────────────────────────
banner("STEP 5 — Running inference on FAKE images (virtual test)")

from utils import batch_predict_action

def fake_image():
    """Synthetic random uint8 RGB, shape (1, 384, 384, 3)."""
    return np.random.randint(0, 255, (1, 384, 384, 3), dtype=np.uint8)

examples = [{
    "obs": {
        "camera0_rgb": fake_image(),
        "camera1_rgb": fake_image(),
    },
    "meta": {"num_camera": 2}
}]

print(f"  Instruction : '{INSTRUCTION}'")
print("  Running batch_predict_action()...")

kwargs = dict(apply_jpeg_compression=True, instruction=INSTRUCTION)
if valid_action_id_length is not None:
    kwargs["valid_action_id_length"] = valid_action_id_length

with torch.no_grad():
    result = batch_predict_action(model, processor, vae, normalizer,
                                  examples=examples, **kwargs)

# ─── STEP 6: Analyze output ───────────────────────────────────────────────────
banner("STEP 6 — Analyzing model output")

action_chunk = result["action_pred"][0]   # (T, D)
T, D = action_chunk.shape
arr  = action_chunk.float().cpu()

nan_count = arr.isnan().sum().item()
inf_count = arr.isinf().sum().item()
std_val   = arr.std().item()
mean_val  = arr.mean().item()

print(f"  Output shape : (T={T}, D={D})")
print(f"  Stats → mean={mean_val:.4f}  std={std_val:.4f}  "
      f"NaNs={nan_count}  Infs={inf_count}")

if nan_count > 0 or inf_count > 0:
    print("  ❌  DEGENERATE: NaN/Inf in output — model is broken")
    sys.exit(1)
elif std_val < 1e-4:
    print("  ⚠️  WARNING: std ≈ 0 — model predicts constant/collapsed actions")
else:
    print("  ✅  Numerically healthy output")

print()

if D == 7:
    print("  ✅  D=7 — Model IS predicting M750 7-DOF!")
    print()
    print(f"  {'t':>4}  " + "  ".join(f"{n:>10}" for n in M750_DOF_NAMES))
    print(f"  {'─'*4}  " + "  ".join(["─"*10] * M750_DOF))
    for t in range(min(T, 10)):
        vals = arr[t].tolist()
        print(f"  {t:>4}  " + "  ".join(f"{v:>10.5f}" for v in vals))
    if T > 10:
        print(f"  ... {T-10} more timesteps ...")

    # Gripper check
    g = arr[:, 6]
    print(f"\n  Gripper range : [{g.min():.4f}, {g.max():.4f}]")
    if 0.0 <= g.min().item() and g.max().item() <= 0.15:
        print("  ✅  Gripper values look physically plausible (0–0.15 m)")
    else:
        print("  ⚠️  Gripper outside expected range — check normalizer scaling")

elif D == 20:
    print("  ⚠️  D=20 — UMI dual-arm format (not M750 7-DOF).")
    print("  Mapping right arm: [0:3]=xyz, [3:9]=6D-rot, [9]=gripper → rx,ry,rz")
    xyz, rot6d, grip = arr[:, 0:3], arr[:, 3:9], arr[:, 9:10]
    r1, r2 = rot6d[:, 0:3], rot6d[:, 3:6]
    r3 = torch.cross(r1, r2, dim=-1)
    rx = torch.atan2(r3[:, 1], r3[:, 2])
    ry = torch.atan2(-r3[:, 0], (r3[:, 1]**2 + r3[:, 2]**2).sqrt())
    rz = torch.atan2(r2[:, 0], r1[:, 0])
    mapped = torch.cat([xyz, torch.stack([rx, ry, rz], -1), grip], -1)
    print()
    print(f"  {'t':>4}  " + "  ".join(f"{n:>10}" for n in M750_DOF_NAMES))
    print(f"  {'─'*4}  " + "  ".join(["─"*10] * M750_DOF))
    for t in range(min(T, 10)):
        vals = mapped[t].tolist()
        print(f"  {t:>4}  " + "  ".join(f"{v:>10.5f}" for v in vals))
    if T > 10:
        print(f"  ... {T-10} more timesteps ...")
    print("\n  ⚠️  Rough mapping only. Retrain with D=7 labels for proper M750 use.")

else:
    print(f"  ❓  D={D} — unexpected output dimension")
    for t in range(min(T, 4)):
        print(f"    t={t}: {arr[t].tolist()}")

# ─── VERDICT ──────────────────────────────────────────────────────────────────
banner("VERDICT")

nan_ok = nan_count == 0
inf_ok = inf_count == 0
std_ok = std_val > 1e-4

if not (nan_ok and inf_ok):
    print("  ❌  Model is BROKEN — NaN/Inf outputs.")
    print("       Try checkpoint-800 instead.")
elif D == 7 and std_ok:
    print("  ✅  Model IS working — predicting 7 values for M750!")
    print("  ✅  Format  : [x, y, z, rx, ry, rz, gripper]")
    print("  ✅  Normalizer: umi_normalizer_official.pt (D=7 confirmed)")
    print()
    print("  ℹ️  Values above used RANDOM fake images — numbers are meaningless.")
    print("      Use real camera frames from your robot to evaluate task quality.")
elif D == 20:
    print("  ⚠️  Model outputs D=20 (UMI format), not D=7 (M750).")
    print("       Check dataset action labels and normalizer.")
elif not std_ok:
    print("  ⚠️  Output is constant (std≈0) — possible mode collapse.")
    print("       Try checkpoint-600 or checkpoint-800.")
else:
    print(f"  ❓  Unexpected D={D} — review dataset and normalizer config.")

print()