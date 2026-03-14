"""
STEP 2: Train VQ-VAE for D=7 — FIXED (no vq loss explosion)
=============================================================
Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python step2_train_vqvae.py
"""

import os, sys, glob, tarfile, io, torch, shutil
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

SHARDS_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/shards"
OUTPUT_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-7dof"
RDT2_DIR    = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
ACTION_DIM  = 7
ACTION_HZ   = 24
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-4
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, RDT2_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Device : {DEVICE}")

import torch.distributed as dist
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29503"
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
print("  ✅ Process group initialized")

# ── PATCH multivqvae.py ───────────────────────────────────────────────────────
print("\n[1/5] Patching multivqvae.py...")
VQVAE_FILE = f"{RDT2_DIR}/vqvae/models/multivqvae.py"
if not os.path.exists(VQVAE_FILE + ".bak_original"):
    shutil.copy(VQVAE_FILE, VQVAE_FILE + ".bak_original")

with open(VQVAE_FILE, "w") as fh:
    fh.write('''"""multivqvae.py — M750 D=7 patch"""
import torch, torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from vqvae.models.vqvae import VQVAE

def _sel(x, k):
    if k=="pos":  return x[...,0:3]
    if k=="rot":  return x[...,3:6]
    if k=="grip": return x[...,6:7]

class MultiVQVAE(nn.Module, PyTorchModelHubMixin):
    def __init__(self,input_dim,embedding_dim,cnn_config,
                 num_embeddings,action_horizon,
                 n_codebooks=None,codebook_restart_interval=64,
                 commitment_cost=0.25,codebook_cost=0.,local_rank=0):
        super().__init__()
        if n_codebooks is None: n_codebooks={"pos":6,"rot":2,"grip":1}
        kw=dict(embedding_dim=embedding_dim,cnn_config=cnn_config,
                num_embeddings=num_embeddings,action_horizon=action_horizon,
                codebook_restart_interval=codebook_restart_interval,
                commitment_cost=commitment_cost,codebook_cost=codebook_cost,
                local_rank=local_rank)
        self.pos_vqvae =VQVAE(input_dim=input_dim["pos"], n_codebooks=n_codebooks["pos"], **kw)
        self.rot_vqvae =VQVAE(input_dim=input_dim["rot"], n_codebooks=n_codebooks["rot"], **kw)
        self.grip_vqvae=VQVAE(input_dim=input_dim["grip"],n_codebooks=n_codebooks["grip"],**kw)
        self.pos_id_len =n_codebooks["pos"]*3
        self.rot_id_len =n_codebooks["rot"]*3
        self.grip_id_len=n_codebooks["grip"]*3
        self.action_dim =input_dim["pos"]+input_dim["rot"]+input_dim["grip"]
        self.action_horizon=action_horizon
        self.num_embeddings=num_embeddings

    def encode(self,x):
        return torch.cat([self.pos_vqvae.encode(_sel(x,"pos")),
                          self.rot_vqvae.encode(_sel(x,"rot")),
                          self.grip_vqvae.encode(_sel(x,"grip"))],dim=-1)

    def decode(self,ids,return_dict=False):
        p=self.pos_vqvae.decode(ids[...,:self.pos_id_len])
        r=self.rot_vqvae.decode(ids[...,self.pos_id_len:self.pos_id_len+self.rot_id_len])
        g=self.grip_vqvae.decode(ids[...,self.pos_id_len+self.rot_id_len:])
        if return_dict: return {"pos":p,"rot":r,"grip":g}
        out=torch.zeros(ids.shape[0],self.action_horizon,self.action_dim,device=ids.device)
        out[...,0:3]=p; out[...,3:6]=r; out[...,6:7]=g
        return out
''')
print("  ✅ patched")

# ── NORMALIZER ────────────────────────────────────────────────────────────────
print("\n[2/5] Computing normalization stats...")

all_actions = []
tar_files = sorted(glob.glob(f"{SHARDS_DIR}/*.tar"))
for tar_path in tar_files:
    try:
        with tarfile.open(tar_path, "r") as tar:
            for m in tar.getmembers():
                if not m.name.endswith(".action.npy"): continue
                raw = tar.extractfile(m).read()
                arr = np.load(io.BytesIO(raw)).astype(np.float32)
                if arr.shape == (ACTION_HZ, ACTION_DIM):
                    all_actions.append(arr.reshape(-1, ACTION_DIM))
    except: pass

all_actions = np.concatenate(all_actions, axis=0)
act_mean = torch.tensor(all_actions.mean(0), dtype=torch.float32)
act_std  = torch.clamp(torch.tensor(all_actions.std(0), dtype=torch.float32), min=1e-6)

norm_path = "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt"
os.makedirs(os.path.dirname(norm_path), exist_ok=True)
torch.save({"mean": act_mean, "std": act_std}, norm_path)
print(f"  ✅ Normalizer saved: {norm_path}")

act_mean_dev = act_mean.to(DEVICE)
act_std_dev  = act_std.to(DEVICE)

# ── DATASET ───────────────────────────────────────────────────────────────────
print("\n[3/5] Loading dataset (normalized)...")

class M750TarDataset(Dataset):
    def __init__(self, shards_dir):
        self.actions = []
        for tar_path in sorted(glob.glob(f"{shards_dir}/*.tar")):
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for m in tar.getmembers():
                        if not m.name.endswith(".action.npy"): continue
                        raw = tar.extractfile(m).read()
                        arr = np.load(io.BytesIO(raw)).astype(np.float32)
                        if arr.shape == (ACTION_HZ, ACTION_DIM):
                            # normalize to mean=0 std=1
                            arr = (arr - act_mean.numpy()) / act_std.numpy()
                            self.actions.append(torch.from_numpy(arr))
            except: pass
        print(f"  Loaded: {len(self.actions)} chunks (normalized)")
        # Verify normalization worked
        sample = torch.stack(self.actions[:1000]).reshape(-1, ACTION_DIM)
        print(f"  After norm — mean≈{sample.mean():.3f} std≈{sample.std():.3f}  (expect ~0, ~1)")

    def __len__(self): return len(self.actions)
    def __getitem__(self, i): return self.actions[i]

dataset = M750TarDataset(SHARDS_DIR)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=4, pin_memory=True, drop_last=True)

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
print("\n[4/5] Building VQ-VAE...")
from vqvae.models.multivqvae import MultiVQVAE

VQVAE_CONFIG = {
    "input_dim":     {"pos": 3, "rot": 3, "grip": 1},
    "embedding_dim": 64,
    "cnn_config":    {"output_size": 64, "hidden_size": 128, "dropout": 0.1},
    "num_embeddings": 512,
    "action_horizon": ACTION_HZ,
    "n_codebooks":    {"pos": 6, "rot": 2, "grip": 1},
    "local_rank":     0,
}
vqvae = MultiVQVAE(**VQVAE_CONFIG).to(DEVICE)
total_id = vqvae.pos_id_len + vqvae.rot_id_len + vqvae.grip_id_len
print(f"  action_dim={vqvae.action_dim}, total_ids={total_id}")

# Debug: check what forward() actually returns in loss_dict
with torch.no_grad():
    test = torch.randn(2, ACTION_HZ, 3).to(DEVICE)
    recon, loss_dict, _ = vqvae.pos_vqvae(test)
    print(f"\n  loss_dict keys : {list(loss_dict.keys())}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.6f}")

print("  ✅ Model ready")

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Training for {EPOCHS} epochs...")
print(f"  LR={LR}, normalized input → expect loss to drop to <0.1")

optimizer = torch.optim.Adam(vqvae.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)
best_loss = float("inf")

for epoch in range(EPOCHS):
    vqvae.train()
    tot_recon = 0.; tot_vq = 0.; n = 0

    for batch in loader:
        batch = batch.to(DEVICE)   # (B, 24, 7) normalized

        # forward() returns (x_recon, loss_dict, indices)
        pos_recon,  pos_ld,  _ = vqvae.pos_vqvae( batch[..., 0:3])
        rot_recon,  rot_ld,  _ = vqvae.rot_vqvae( batch[..., 3:6])
        grip_recon, grip_ld, _ = vqvae.grip_vqvae(batch[..., 6:7])

        # Pure MSE reconstruction — no VQ loss (avoids explosion)
        recon_loss = (F.mse_loss(pos_recon,  batch[..., 0:3]) +
                      F.mse_loss(rot_recon,  batch[..., 3:6]) +
                      F.mse_loss(grip_recon, batch[..., 6:7]))

        # VQ commitment loss — use small weight 0.1 to avoid explosion
        # loss_dict has 'vq' key but values can be large — scale down
        vq_loss = 0.1 * (pos_ld['commitment'] + rot_ld['commitment'] + grip_ld['commitment']) \
                  if 'commitment' in pos_ld else \
                  0.01 * (pos_ld['vq'] + rot_ld['vq'] + grip_ld['vq'])

        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vqvae.parameters(), 1.0)
        optimizer.step()

        tot_recon += recon_loss.item()
        tot_vq    += vq_loss.item()
        n += 1

    scheduler.step()
    avg_recon = tot_recon / n
    avg_vq    = tot_vq / n
    avg_total = avg_recon + avg_vq

    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1:>3}/{EPOCHS} | "
              f"total={avg_total:.4f} | "
              f"recon={avg_recon:.4f} | "
              f"vq={avg_vq:.4f}")

    if (epoch+1) % 25 == 0:
        path = f"{OUTPUT_DIR}/vqvae_epoch{epoch+1}.pt"
        torch.save({"epoch": epoch+1, "model": vqvae.state_dict(),
                    "config": VQVAE_CONFIG, "loss": avg_total,
                    "norm": {"mean": act_mean, "std": act_std}}, path)
        print(f"  💾 {path}")

    if avg_total < best_loss:
        best_loss = avg_total
        torch.save({"epoch": epoch+1, "model": vqvae.state_dict(),
                    "config": VQVAE_CONFIG, "loss": best_loss,
                    "norm": {"mean": act_mean, "std": act_std}},
                   f"{OUTPUT_DIR}/vqvae_best.pt")

torch.save({"epoch": EPOCHS, "model": vqvae.state_dict(),
            "config": VQVAE_CONFIG, "loss": avg_total,
            "norm": {"mean": act_mean, "std": act_std}},
           f"{OUTPUT_DIR}/vqvae_final.pt")

print(f"\n✅ Done! Best loss: {best_loss:.4f}")
print(f"   Saved: {OUTPUT_DIR}/vqvae_final.pt")
print(f"\nNext: python step3_build_normalizer.py")