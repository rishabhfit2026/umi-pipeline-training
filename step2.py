"""
STEP 2: Re-tokenize all shards with fixed VQ-VAE
=================================================
Replaces the all-zero action_token.npy with correct tokens.

Run AFTER step1_train_vqvae_fixed.py:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python step2_retokenize_shards.py
"""

import os, sys, glob, tarfile, io, torch, shutil
import numpy as np
from tqdm import tqdm

SHARDS_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/shards"
VQVAE_CKPT  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-fixed/vqvae_best.pt"  # FIXED: best not final
RDT2_DIR    = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
ACTION_HZ   = 24
ACTION_DIM  = 7

sys.path.insert(0, RDT2_DIR)

import torch.distributed as dist
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29506"
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

print("="*60)
print("  Step 2: Re-tokenizing shards with fixed VQ-VAE")
print("="*60)
print(f"  Using checkpoint: vqvae_best.pt")

# ── LOAD VQ-VAE ───────────────────────────────────────────────────────────────
print("\n[1/3] Loading VQ-VAE...")
from vqvae.models.multivqvae import MultiVQVAE

ckpt  = torch.load(VQVAE_CKPT, map_location="cpu")
vqvae = MultiVQVAE(**ckpt["config"]).eval().to(DEVICE)
vqvae.load_state_dict(ckpt["model"])

act_mean    = ckpt["norm"]["mean"].to(DEVICE)
act_std     = ckpt["norm"]["std"].to(DEVICE)
act_mean_np = ckpt["norm"]["mean"].numpy()
act_std_np  = ckpt["norm"]["std"].numpy()

total_ids = vqvae.pos_id_len + vqvae.rot_id_len + vqvae.grip_id_len
print(f"  Epoch: {ckpt['epoch']}  Loss: {ckpt['loss']:.4f}")
print(f"  total_ids={total_ids}")

# Quick verify diversity before we start
with torch.no_grad():
    results = []
    for seed in [0, 42, 99]:
        torch.manual_seed(seed)
        ids = torch.randint(0, 512, (1, total_ids)).to(DEVICE)
        out = vqvae.decode(ids).squeeze(0)
        out_real = (out * act_std + act_mean).cpu().numpy()
        results.append(round(out_real[0, 0] * 1000, 1))
    if len(set(results)) == 1:
        print("  ❌ VQ-VAE decode still collapsed! Run step1 again.")
        sys.exit(1)
    else:
        print(f"  ✅ VQ-VAE decode is diverse: {results} mm")

# ── RETOKENIZE ────────────────────────────────────────────────────────────────
print("\n[2/3] Re-tokenizing all shards...")

tar_files = sorted(glob.glob(f"{SHARDS_DIR}/shard-*.tar"))
print(f"  Found {len(tar_files)} shards")

total_samples         = 0
token_diversity_check = []

for tar_path in tqdm(tar_files, desc="Shards"):
    tmp_path = tar_path + ".tmp"

    with tarfile.open(tar_path, "r") as tar_in:
        members = {m.name: m for m in tar_in.getmembers()}

        with tarfile.open(tmp_path, "w") as tar_out:
            for name, member in members.items():
                data = tar_in.extractfile(member)
                if data is None:
                    continue

                if name.endswith(".action_token.npy"):
                    prefix      = name.replace(".action_token.npy", "")
                    action_name = f"{prefix}.action.npy"

                    if action_name in members:
                        act_data = tar_in.extractfile(members[action_name]).read()
                        act      = np.load(io.BytesIO(act_data)).astype(np.float32)

                        if act.shape == (ACTION_HZ, ACTION_DIM):
                            act_norm = (act - act_mean_np) / act_std_np
                            act_t    = torch.tensor(act_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

                            with torch.no_grad():
                                token_ids = vqvae.encode(act_t).squeeze(0).cpu().numpy().astype(np.int16)

                            total_samples += 1
                            if total_samples <= 100:
                                token_diversity_check.append(token_ids)

                            buf = io.BytesIO()
                            np.save(buf, token_ids)
                            buf.seek(0)
                            tok_bytes = buf.read()

                            info      = tarfile.TarInfo(name=name)
                            info.size = len(tok_bytes)
                            tar_out.addfile(info, io.BytesIO(tok_bytes))
                            continue

                # Copy all other files unchanged
                raw       = data.read()
                info      = tarfile.TarInfo(name=name)
                info.size = len(raw)
                tar_out.addfile(info, io.BytesIO(raw))

    shutil.move(tmp_path, tar_path)

# ── VERIFY ────────────────────────────────────────────────────────────────────
print(f"\n[3/3] Verification...")
print(f"  Total samples re-tokenized: {total_samples}")

if token_diversity_check:
    all_toks     = np.stack(token_diversity_check)
    unique       = len(np.unique(all_toks))
    sample_means = all_toks.mean(axis=1)
    print(f"  Unique tokens in first 100 samples: {unique}/512")
    print(f"  Token range: [{all_toks.min()}, {all_toks.max()}]")
    print(f"  Token mean per sample (should vary): {sample_means[:5].round(1)}")

    if unique < 10:
        print("  ❌ Still collapsed! Check VQ-VAE training.")
    elif unique > 50:
        print("  ✅ Good token diversity!")
    else:
        print("  ⚠️  Low diversity - VQ-VAE may need more training")

# Decode check on first sample
print("\n  Decode check on first re-tokenized sample:")
t       = tarfile.open(tar_files[0])
members = {m.name: m for m in t.getmembers()}
tok     = np.load(io.BytesIO(t.extractfile(members['000000.action_token.npy']).read()))
act     = np.load(io.BytesIO(t.extractfile(members['000000.action.npy']).read()))
t.close()

ids = torch.tensor(tok).unsqueeze(0).long().to(DEVICE)
with torch.no_grad():
    decoded_norm = vqvae.decode(ids).squeeze(0)
    decoded      = (decoded_norm * act_std + act_mean).cpu().numpy()

err = np.abs(act - decoded).mean()
print(f"  Real    action[0]: {act[0].round(3)}")
print(f"  Decoded action[0]: {decoded[0].round(3)}")
print(f"  Reconstruction error: {err:.4f}")

if err < 0.05:
    print("  ✅ Excellent reconstruction!")
elif err < 0.1:
    print("  ✅ Good reconstruction")
else:
    print("  ⚠️  High error — may need more VQ-VAE training")

print("\n✅ Re-tokenization complete!")
print("   Next: python step3_finetune_rdt2.py")