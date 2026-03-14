"""
STEP 3 (rewrite): Fine-tune RDT2 directly from zarr data
=========================================================
- Reads directly from replay_buffer.zarr (no shards needed)
- Uses per-DOF binning (256 bins each) — structured tokens, model will learn
- 7 tokens per frame: eef_pos(3) + eef_rot(3) + gripper(1)
- Predicts next N frames given current image

Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python step3_zarr_rdt2.py
"""

import os, sys, torch, zarr, numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ── CONFIG ────────────────────────────────────────────────────────────────────
ZARR_PATH  = "/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr"
OUTPUT_DIR = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-zarr"
BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

N_BINS     = 256      # bins per DOF
N_DOFS     = 7        # eef_pos(3) + eef_rot(3) + gripper(1)
PRED_STEPS = 8        # predict next 8 frames = 56 tokens total
BATCH_SIZE = 2
GRAD_ACCUM = 8
MAX_STEPS  = 3000
SAVE_STEPS = 500
LR_LORA    = 2e-4
LR_HEAD    = 1e-3
LORA_R     = 32
LORA_ALPHA = 64
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

INSTRUCTION = "pick up the marker and place it in the box"

# Token layout: DOF d, bin b → token_id = d * N_BINS + b
# dof 0=eef_x, 1=eef_y, 2=eef_z, 3=rot_x, 4=rot_y, 5=rot_z, 6=gripper
N_ACT_TOKENS = N_DOFS * N_BINS   # 1792

os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, "/home/rishabh/Downloads/umi-pipeline-training/RDT2")

print("="*60)
print("  RDT2 from Zarr — Per-DOF Binning (RT-2 style)")
print("="*60)
print(f"  {N_DOFS} DOFs × {N_BINS} bins = {N_ACT_TOKENS} action tokens")
print(f"  Predicting {PRED_STEPS} frames = {PRED_STEPS*N_DOFS} tokens per sample")

# ── COMPUTE BIN EDGES FROM DATA ───────────────────────────────────────────────
print("\n[1/5] Computing bin edges from zarr data...")
z        = zarr.open(ZARR_PATH, "r")
eef_pos  = z["data"]["robot0_eef_pos"][:]          # (N,3)
eef_rot  = z["data"]["robot0_eef_rot_axis_angle"][:] # (N,3)
grip     = z["data"]["robot0_gripper_width"][:]    # (N,1)
ep_ends  = z["meta"]["episode_ends"][:]

actions  = np.concatenate([eef_pos, eef_rot, grip], axis=1)  # (N,7)
print(f"  Total frames: {len(actions)}  Episodes: {len(ep_ends)}")

# Per-DOF bin edges using percentiles (handles outliers better than min/max)
bin_edges = []
for d in range(N_DOFS):
    vals = actions[:, d]
    edges = np.percentile(vals, np.linspace(0, 100, N_BINS + 1))
    edges[0]  -= 1e-6
    edges[-1] += 1e-6
    bin_edges.append(edges)
    print(f"  DOF {d}: range=[{edges[0]:.3f}, {edges[-1]:.3f}]")

bin_edges = np.stack(bin_edges)  # (7, 257)
np.save(f"{OUTPUT_DIR}/bin_edges.npy", bin_edges)

def encode_action(action_7dof):
    """Convert (7,) float action → (7,) token ids (0..1791)"""
    tokens = np.zeros(N_DOFS, dtype=np.int32)
    for d in range(N_DOFS):
        b = np.searchsorted(bin_edges[d], action_7dof[d], side='right') - 1
        tokens[d] = d * N_BINS + np.clip(b, 0, N_BINS - 1)
    return tokens

def decode_action(tokens_7):
    """Convert (7,) token ids → (7,) float action (bin centers)"""
    action = np.zeros(N_DOFS)
    for d in range(N_DOFS):
        b = tokens_7[d] - d * N_BINS
        b = np.clip(b, 0, N_BINS - 1)
        action[d] = (bin_edges[d][b] + bin_edges[d][b+1]) / 2
    return action

# Verify encoding roundtrip
test_action = actions[100]
encoded     = encode_action(test_action)
decoded     = decode_action(encoded)
err         = np.abs(test_action - decoded).max()
print(f"\n  Roundtrip check: max_err={err:.5f}  ({'✅' if err < 0.01 else '❌'})")
print(f"  Sample tokens: {encoded.tolist()}")

# Check correlation between consecutive tokens (should be high now)
sample_tokens = np.stack([encode_action(actions[i]) for i in range(0, 1000, 1)])
corrs = [np.corrcoef(sample_tokens[:,d], sample_tokens[:,d])[0,1] for d in range(6)]
cross_corrs = [np.corrcoef(sample_tokens[:-1,d], sample_tokens[1:,d])[0,1] for d in range(6)]
print(f"  Temporal autocorrelation (consecutive frames, same DOF): {np.mean(cross_corrs):.4f}")
print(f"  {'✅ Structured!' if np.mean(cross_corrs) > 0.5 else '⚠️  Low — check data'}")

# ── DATASET ───────────────────────────────────────────────────────────────────
print("\n[2/5] Building dataset...")

class ZarrDataset(Dataset):
    def __init__(self):
        self.samples = []
        imgs    = z["data"]["camera0_rgb"]
        ep_starts = np.concatenate([[0], ep_ends[:-1]])

        for ep_idx in range(len(ep_ends)):
            start = ep_starts[ep_idx]
            end   = ep_ends[ep_idx]
            # Sample every 4 frames as observation point
            for obs_idx in range(start, end - PRED_STEPS * 4, 4):
                # Future action frames (every 4 frames = ~0.13s at 30fps)
                future_idxs = [obs_idx + (k+1)*4 for k in range(PRED_STEPS)]
                if future_idxs[-1] >= end:
                    continue
                # Get image at obs_idx
                img_np = imgs[obs_idx]  # (224,224,3)
                img    = Image.fromarray(img_np)
                # Get future action tokens
                act_tokens = []
                for fi in future_idxs:
                    tok = encode_action(actions[fi])
                    act_tokens.extend(tok.tolist())
                self.samples.append({
                    "image":      img,
                    "act_tokens": np.array(act_tokens, dtype=np.int32),  # (PRED_STEPS*7,)
                })

        print(f"  Built {len(self.samples)} samples from {len(ep_ends)} episodes")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

dataset = ZarrDataset()
SEQ_LEN = PRED_STEPS * N_DOFS  # 56 tokens per sample

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("\n[3/5] Loading model...")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True)

# Add N_ACT_TOKENS=1792 new tokens
new_tokens = [f"<act_{d}_{b}>" for d in range(N_DOFS) for b in range(N_BINS)]
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": new_tokens},
    replace_additional_special_tokens=False)

# Build lookup: (dof, bin) → vocab_id
ACT_VOCAB_START = processor.tokenizer.convert_tokens_to_ids("<act_0_0>")
ACT_SET = set(range(ACT_VOCAB_START, ACT_VOCAB_START + N_ACT_TOKENS))

def token_id(dof, bin_idx):
    return ACT_VOCAB_START + dof * N_BINS + bin_idx

def vocab_to_dof_bin(vocab_id):
    offset = vocab_id - ACT_VOCAB_START
    return offset // N_BINS, offset % N_BINS

VOCAB_SIZE = len(processor.tokenizer)
print(f"  Vocab: {VOCAB_SIZE}  act tokens: {ACT_VOCAB_START}..{ACT_VOCAB_START+N_ACT_TOKENS-1}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager")
model.resize_token_embeddings(VOCAB_SIZE)

lora_cfg = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    init_lora_weights="gaussian")
model = get_peft_model(model, lora_cfg).to(DEVICE)

# Direct access to lm_head and embed_tokens
lm_head_w   = model.base_model.model.lm_head.weight
embed_tok_w = model.base_model.model.model.embed_tokens.weight
lm_head_w.requires_grad   = True
embed_tok_w.requires_grad = True

lora_p = sum(p.numel() for p in model.parameters()
             if p.requires_grad
             and p.data_ptr() != lm_head_w.data_ptr()
             and p.data_ptr() != embed_tok_w.data_ptr())
print(f"  LoRA params: {lora_p:,}")
print(f"  Head params: {lm_head_w.numel() + embed_tok_w.numel():,}")
print("  ✅ Model ready")

# ── COLLATE ───────────────────────────────────────────────────────────────────
def collate(batch):
    results = []
    for s in batch:
        act_tokens = s["act_tokens"]  # (SEQ_LEN,) values are global token offsets
        img        = s["image"]

        # Build token string: <act_d_b> for each (dof, bin)
        act_str = ""
        for i, tok_offset in enumerate(act_tokens):
            d = (i % N_DOFS)
            b = tok_offset - d * N_BINS
            act_str += f"<act_{d}_{b}>"

        messages = [
            {"role": "user",      "content": [{"type": "image"},
                                               {"type": "text", "text": INSTRUCTION}]},
            {"role": "assistant", "content": [{"type": "text", "text": act_str}]}
        ]
        text   = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(text=[text], images=[[img]], return_tensors="pt", padding=True)
        ids    = inputs["input_ids"][0]

        # Labels: -100 everywhere except action token positions
        labels = torch.full_like(ids, -100)
        act_count = 0
        for pos in range(len(ids)):
            if ids[pos].item() in ACT_SET:
                labels[pos] = ids[pos]
                act_count   += 1

        if act_count != SEQ_LEN:
            continue

        results.append({
            "input_ids":      ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values":   inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels":         labels,
        })
    return results

# Verify
print("\n  Verifying collate...")
t = collate([dataset[0]]); assert t, "Collate failed!"
s = t[0]
n = (s["labels"] != -100).sum().item()
print(f"  seq_len={len(s['input_ids'])}  action_labels={n}  (want {SEQ_LEN})")
assert n == SEQ_LEN, f"Expected {SEQ_LEN} action labels, got {n}"

# Init loss check
model.eval()
with torch.no_grad():
    kw  = {"input_ids": s["input_ids"].unsqueeze(0).to(DEVICE),
           "attention_mask": s["attention_mask"].unsqueeze(0).to(DEVICE)}
    if s["pixel_values"] is not None:
        kw["pixel_values"]   = s["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
        kw["image_grid_thw"] = s["image_grid_thw"].to(DEVICE)
    out     = model(**kw)
    lbl_cpu = s["labels"]
    act_pos = (lbl_cpu != -100).nonzero(as_tuple=True)[0]
    pred_p  = (act_pos - 1).to(DEVICE)
    # For each action token, only score over its DOF's 256 bins
    total_loss = 0.0
    for i, (pp, ap) in enumerate(zip(pred_p, act_pos)):
        dof = i % N_DOFS
        start = ACT_VOCAB_START + dof * N_BINS
        logit = out.logits[0][pp, start:start+N_BINS].float()
        tgt   = (lbl_cpu[ap].item() - start)
        total_loss += F.cross_entropy(logit.unsqueeze(0), torch.tensor([tgt]).to(DEVICE)).item()
    il = total_loss / SEQ_LEN
    print(f"  Init loss: {il:.4f}  (ln256={np.log(256):.4f})")
    print(f"  {'✅' if abs(il - np.log(256)) < 0.5 else '⚠️  unexpected init loss'}")
model.train()

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training {MAX_STEPS} steps...")

lora_params = [p for p in model.parameters()
               if p.requires_grad
               and p.data_ptr() != lm_head_w.data_ptr()
               and p.data_ptr() != embed_tok_w.data_ptr()]
head_params = [lm_head_w, embed_tok_w]

optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": LR_LORA},
    {"params": head_params, "lr": LR_HEAD},
], weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=[LR_LORA, LR_HEAD],
    total_steps=MAX_STEPS, pct_start=0.05, anneal_strategy='cos')

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate, num_workers=0, drop_last=True)

model.train()
step = 0; running_loss = 0.0
optimizer.zero_grad()

for epoch in range(1000):
    for batch in loader:
        if step >= MAX_STEPS: break
        if not batch: continue

        total_loss = torch.tensor(0.0, device=DEVICE)
        valid = 0

        for s in batch:
            try:
                kw = {"input_ids":      s["input_ids"].unsqueeze(0).to(DEVICE),
                      "attention_mask": s["attention_mask"].unsqueeze(0).to(DEVICE)}
                if s["pixel_values"] is not None:
                    kw["pixel_values"]   = s["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
                    kw["image_grid_thw"] = s["image_grid_thw"].to(DEVICE)

                out     = model(**kw)
                logits  = out.logits[0]
                lbl_cpu = s["labels"]
                act_pos = (lbl_cpu != -100).nonzero(as_tuple=True)[0]
                pred_p  = (act_pos - 1).to(DEVICE)

                # Per-DOF cross entropy (256-way classification each)
                loss = torch.tensor(0.0, device=DEVICE)
                for i, (pp, ap) in enumerate(zip(pred_p, act_pos)):
                    dof   = i % N_DOFS
                    start = ACT_VOCAB_START + dof * N_BINS
                    logit = logits[pp, start:start+N_BINS].float()
                    tgt   = torch.tensor([lbl_cpu[ap].item() - start], device=DEVICE)
                    loss  = loss + F.cross_entropy(logit.unsqueeze(0), tgt)
                loss = loss / SEQ_LEN

                if not loss.isnan():
                    total_loss += loss; valid += 1
            except Exception as e:
                torch.cuda.empty_cache()

        if valid > 0:
            (total_loss / valid / GRAD_ACCUM).backward()
            running_loss += (total_loss / valid).item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        step += 1
        if step % 50 == 0:
            avg = running_loss / 50; running_loss = 0.0
            tgt_loss = np.log(256)
            st  = "✅" if avg < tgt_loss * 0.5 else "⚠️ " if avg < tgt_loss * 0.8 else "❌"
            print(f"  Step {step:>5}/{MAX_STEPS} | loss={avg:.4f} {st} | lr={scheduler.get_last_lr()[0]:.2e}")

        if step % SAVE_STEPS == 0:
            d = f"{OUTPUT_DIR}/checkpoint-{step}"
            os.makedirs(d, exist_ok=True)
            model.save_pretrained(d); processor.save_pretrained(d)
            np.save(f"{d}/bin_edges.npy", bin_edges)
            torch.save({"act_vocab_start": ACT_VOCAB_START,
                        "n_bins": N_BINS, "n_dofs": N_DOFS,
                        "pred_steps": PRED_STEPS}, f"{d}/action_config.pt")
            print(f"  💾 Saved: {d}")

    if step >= MAX_STEPS: break

d = f"{OUTPUT_DIR}/checkpoint-final"
os.makedirs(d, exist_ok=True)
model.save_pretrained(d); processor.save_pretrained(d)
np.save(f"{d}/bin_edges.npy", bin_edges)
torch.save({"act_vocab_start": ACT_VOCAB_START,
            "n_bins": N_BINS, "n_dofs": N_DOFS,
            "pred_steps": PRED_STEPS}, f"{d}/action_config.pt")
print(f"\n✅ Training done! → {d}")
print(f"   Next: python step4_maniskill_inference.py")