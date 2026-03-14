"""
STEP 1: Train VQ-VAE FIXED v4
==============================
Fixes applied:
  - LR = 3e-4  (v3 had 1e-3 which caused recon=26 divergence)
  - count_unique_tokens uses vqvae.encode() correctly
  - vq.py patches: ema_decay 0.99→0.95, OOB wrap, ema_cluster_size tracking
  - commitment_cost=1.0, codebook_restart_interval=32

Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python step1_train_vqvae_fixed.py
"""

import os, sys, glob, tarfile, io, torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

SHARDS_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/shards"
OUTPUT_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-fixed"
RDT2_DIR    = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
NORM_PATH   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt"
ACTION_DIM  = 7
ACTION_HZ   = 24
BATCH_SIZE  = 128
EPOCHS      = 150
LR          = 3e-4      # FIXED: 1e-3 caused recon=26 divergence
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, RDT2_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("  VQ-VAE Training v4 — LR Fixed (3e-4)")
print("="*60)
print(f"  Device: {DEVICE}")

import torch.distributed as dist
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("\n[1/4] Loading data...")
all_actions = []
for tar_path in sorted(glob.glob(f"{SHARDS_DIR}/*.tar")):
    try:
        with tarfile.open(tar_path, "r") as tar:
            for m in tar.getmembers():
                if not m.name.endswith(".action.npy"): continue
                arr = np.load(io.BytesIO(tar.extractfile(m).read())).astype(np.float32)
                if arr.shape == (ACTION_HZ, ACTION_DIM):
                    all_actions.append(arr)
    except: pass

all_actions = np.stack(all_actions)
flat        = all_actions.reshape(-1, ACTION_DIM)
act_mean    = torch.tensor(flat.mean(0), dtype=torch.float32)
act_std     = torch.clamp(torch.tensor(flat.std(0), dtype=torch.float32), min=1e-6)
act_mean_np = act_mean.numpy()
act_std_np  = act_std.numpy()

print(f"  {len(all_actions)} samples loaded")
print(f"  Mean: {act_mean_np.round(3)}")
print(f"  Std:  {act_std_np.round(3)}")
torch.save({"mean": act_mean, "std": act_std}, NORM_PATH)
print(f"  Normalizer saved: {NORM_PATH}")

class ActionDataset(Dataset):
    def __init__(self):
        normed = (all_actions - act_mean_np) / act_std_np
        self.data = torch.tensor(normed, dtype=torch.float32)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

dataset = ActionDataset()
loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                     shuffle=True, num_workers=4, drop_last=True)

sample_flat  = dataset.data.reshape(-1, ACTION_DIM)
per_dof_mean = sample_flat.mean(0).numpy().round(3)
per_dof_std  = sample_flat.std(0).numpy().round(3)
print(f"  Per-DOF normed mean: {per_dof_mean}  (want all ~0)")
print(f"  Per-DOF normed std:  {per_dof_std}   (want all ~1)")

# ── MODEL ─────────────────────────────────────────────────────────────────────
print("\n[2/4] Building VQ-VAE...")
from vqvae.models.multivqvae import MultiVQVAE

VQVAE_CONFIG = {
    "input_dim":                 {"pos": 3, "rot": 3, "grip": 1},
    "embedding_dim":             64,
    "cnn_config":                {"output_size": 64, "hidden_size": 128, "dropout": 0.1},
    "num_embeddings":            512,
    "action_horizon":            ACTION_HZ,
    "n_codebooks":               {"pos": 6, "rot": 2, "grip": 1},
    "commitment_cost":           1.0,
    "codebook_restart_interval": 32,
    "local_rank":                0,
}
vqvae     = MultiVQVAE(**VQVAE_CONFIG).to(DEVICE)
total_ids = vqvae.pos_id_len + vqvae.rot_id_len + vqvae.grip_id_len
print(f"  pos_id_len={vqvae.pos_id_len}  rot_id_len={vqvae.rot_id_len}  grip_id_len={vqvae.grip_id_len}")
print(f"  Total tokens per chunk: {total_ids}")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_vq_loss(ld):
    if 'commitment' in ld: return ld['commitment']
    if 'vq'         in ld: return ld['vq']
    return sum(ld.values())

def count_unique_tokens(vqvae, n_samples=500):
    vqvae.eval()
    all_ids = []
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    with torch.no_grad():
        for i in indices:
            x   = dataset[i].unsqueeze(0).to(DEVICE)
            ids = vqvae.encode(x)   # (1, 27) — correct API
            all_ids.append(ids.cpu())
    all_ids = torch.cat(all_ids, dim=0)
    p = vqvae.pos_id_len
    r = vqvae.rot_id_len
    u_pos  = len(torch.unique(all_ids[:, :p]))
    u_rot  = len(torch.unique(all_ids[:, p:p+r]))
    u_grip = len(torch.unique(all_ids[:, p+r:]))
    vqvae.train()
    return u_pos, u_rot, u_grip

def verify_decode_diversity(vqvae):
    vqvae.eval()
    xs = []
    act_mean_dev = act_mean.to(DEVICE)
    act_std_dev  = act_std.to(DEVICE)
    with torch.no_grad():
        for seed in range(8):
            torch.manual_seed(seed)
            ids = torch.randint(0, 512, (1, total_ids)).to(DEVICE)
            out = vqvae.decode(ids).squeeze(0)
            out_r = (out * act_std_dev + act_mean_dev).cpu().numpy()
            xs.append(round(out_r[0, 0] * 1000, 0))
    vqvae.train()
    return xs, len(set(xs))

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print(f"\n[3/4] Training {EPOCHS} epochs...")
print(f"  LR={LR}  commitment_cost=1.0  ema_decay=0.95  restart_interval=32")
print(f"  Target by epoch 20: recon<0.1  pos>50/512  decode_unique>5/8")

optimizer = torch.optim.Adam(vqvae.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)
best_loss = float("inf")

for epoch in range(EPOCHS):
    vqvae.train()
    tot_recon = 0.; tot_vq = 0.; n = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        pos_recon,  pos_ld,  _ = vqvae.pos_vqvae(batch[..., 0:3])
        rot_recon,  rot_ld,  _ = vqvae.rot_vqvae(batch[..., 3:6])
        grip_recon, grip_ld, _ = vqvae.grip_vqvae(batch[..., 6:7])

        recon = (F.mse_loss(pos_recon,  batch[..., 0:3]) +
                 F.mse_loss(rot_recon,  batch[..., 3:6]) +
                 F.mse_loss(grip_recon, batch[..., 6:7]))
        vq    = 0.25 * (get_vq_loss(pos_ld) +
                        get_vq_loss(rot_ld) +
                        get_vq_loss(grip_ld))
        loss  = recon + vq

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        optimizer.step()
        tot_recon += recon.item(); tot_vq += vq.item(); n += 1

    scheduler.step()
    avg_recon  = tot_recon / n
    avg_vq     = tot_vq    / n
    total_loss = avg_recon + avg_vq

    if (epoch + 1) % 10 == 0:
        u_pos, u_rot, u_grip = count_unique_tokens(vqvae)
        xs, u_decode = verify_decode_diversity(vqvae)
        status = "✅" if u_pos > 50 else "⚠️ " if u_pos > 10 else "❌"
        print(f"  Epoch {epoch+1:>3}/{EPOCHS} | "
              f"recon={avg_recon:.4f} | vq={avg_vq:.4f} | "
              f"pos={u_pos}/512 rot={u_rot}/512 grip={u_grip}/512 {status} | "
              f"decode_unique={u_decode}/8")
        print(f"    x decode: {xs}")
        if u_decode < 3 and epoch >= 20:
            print(f"  ❌ Still collapsed! Check: diff vqvae/models/vq.py vqvae/models/vq.py.bak")

    if total_loss < best_loss:
        best_loss = total_loss
        torch.save({"epoch": epoch+1, "model": vqvae.state_dict(),
                    "config": VQVAE_CONFIG, "loss": best_loss,
                    "norm": {"mean": act_mean, "std": act_std}},
                   f"{OUTPUT_DIR}/vqvae_best.pt")

    if (epoch + 1) % 50 == 0:
        path = f"{OUTPUT_DIR}/vqvae_epoch{epoch+1}.pt"
        torch.save({"epoch": epoch+1, "model": vqvae.state_dict(),
                    "config": VQVAE_CONFIG, "loss": total_loss,
                    "norm": {"mean": act_mean, "std": act_std}}, path)
        print(f"  Checkpoint: {path}")

torch.save({"epoch": EPOCHS, "model": vqvae.state_dict(),
            "config": VQVAE_CONFIG, "loss": best_loss,
            "norm": {"mean": act_mean, "std": act_std}},
           f"{OUTPUT_DIR}/vqvae_final.pt")
print(f"\nTraining done! Best loss: {best_loss:.4f}")

# ── FINAL VERIFY ──────────────────────────────────────────────────────────────
print("\n[4/4] Final diversity check...")
vqvae.eval()
act_mean_dev = act_mean.to(DEVICE)
act_std_dev  = act_std.to(DEVICE)
print(f"  {'SEED':<6} {'X(mm)':>8} {'Y(mm)':>8} {'Z(mm)':>8} {'GRIP(mm)':>10}")
print(f"  {'-'*44}")
with torch.no_grad():
    for seed in [0, 42, 99, 123, 7, 13, 55, 88]:
        torch.manual_seed(seed)
        ids = torch.randint(0, 512, (1, total_ids)).to(DEVICE)
        out = vqvae.decode(ids).squeeze(0)
        r   = (out * act_std_dev + act_mean_dev).cpu().numpy()
        print(f"  {seed:<6} {r[0,0]*1000:>+8.1f} {r[0,1]*1000:>+8.1f} "
              f"{r[0,2]*1000:>+8.1f} {r[0,6]*1000:>+10.2f}")

u_pos, u_rot, u_grip = count_unique_tokens(vqvae, n_samples=1000)
print(f"\n  Token usage: pos={u_pos}/512  rot={u_rot}/512  grip={u_grip}/512")
if u_pos > 100:
    print("  ✅ VQ-VAE working! Proceed to: python step2_retokenize_shards.py")
elif u_pos > 30:
    print("  ⚠️  Partial collapse — usable, but retraining recommended")
else:
    print("  ❌ Still collapsed — check vq.py patches:")
    print("     diff vqvae/models/vq.py vqvae/models/vq.py.bak")