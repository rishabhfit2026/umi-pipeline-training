"""
STEP 4: Fine-tune RDT2 — NATIVE TOKEN EMBEDDING (v4)
=============================================
Continues from checkpoint-final for 6000 more steps.
Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python step4_finetune_rdt2.py
"""

import os, sys, glob, tarfile, io, json, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

RDT2_DIR    = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
SHARDS_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/shards"
VQVAE_CKPT  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-7dof/vqvae_final.pt"
NORM_PATH   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt"

# ── KEY CHANGES v4 ────────────────────────────────────────────────────────────
BASE_MODEL  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v3/checkpoint-final"
OUTPUT_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v4"
MAX_STEPS   = 6000   # 6000 more steps on top of v3
SAVE_STEPS  = 500
LR          = 1e-5   # lower LR for fine-tuning on top of checkpoint
# ─────────────────────────────────────────────────────────────────────────────

INSTRUCTION = "pick the marker and place in the box"
ACTION_DIM  = 7
ACTION_HZ   = 24
VALID_LEN   = 27
VOCAB_SIZE  = 512
BATCH_SIZE  = 2
GRAD_ACCUM  = 8
LORA_R      = 16
LORA_ALPHA  = 32
IMG_SIZE    = 336
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, RDT2_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("  Fine-tuning RDT2 v4 — Resuming from v3 checkpoint")
print("="*60)
print(f"  Base   : {BASE_MODEL}")
print(f"  Output : {OUTPUT_DIR}")
print(f"  Steps  : {MAX_STEPS}")
print(f"  LR     : {LR}")

# ── DISTRIBUTED INIT ──────────────────────────────────────────────────────────
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29508"
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

# ── LOAD VQ-VAE + NORMALIZER ──────────────────────────────────────────────────
print("\n[1/5] Loading VQ-VAE + normalizer...")
norm_data = torch.load(NORM_PATH, map_location="cpu")
act_mean  = norm_data["mean"].to(DEVICE)
act_std   = norm_data["std"].to(DEVICE)

from vqvae.models.multivqvae import MultiVQVAE
vqvae_ckpt = torch.load(VQVAE_CKPT, map_location="cpu")
vqvae = MultiVQVAE(**vqvae_ckpt["config"]).eval().to(DEVICE)
vqvae.load_state_dict(vqvae_ckpt["model"])

@torch.no_grad()
def encode_action(action_np):
    """(24,7) numpy → (27,) int64 token ids"""
    act = torch.from_numpy(action_np).float().unsqueeze(0).to(DEVICE)
    act_norm = (act - act_mean) / act_std
    return vqvae.encode(act_norm).squeeze(0).cpu()

print("  ✅ VQ-VAE ready")

# ── DATASET ───────────────────────────────────────────────────────────────────
print("\n[2/5] Loading dataset...")

TASK_INSTRUCTIONS = {
    "task_0": INSTRUCTION,
    "task_1": INSTRUCTION,
}

class M750Dataset(Dataset):
    def __init__(self, shards_dir):
        self.samples = []
        tar_files = sorted(glob.glob(f"{shards_dir}/*.tar"))
        print(f"  Scanning {len(tar_files)} tar files...")

        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    members = {m.name: m for m in tar.getmembers()}
                    prefixes = sorted(set(
                        n.replace(".action.npy", "")
                        for n in members if n.endswith(".action.npy")))
                    for prefix in prefixes:
                        try:
                            act_raw = tar.extractfile(
                                members[f"{prefix}.action.npy"]).read()
                            action = np.load(
                                io.BytesIO(act_raw)).astype(np.float32)
                            if action.shape != (ACTION_HZ, ACTION_DIM):
                                continue
                            img_raw = tar.extractfile(
                                members[f"{prefix}.image.jpg"]).read()
                            img = Image.open(
                                io.BytesIO(img_raw)).convert("RGB")
                            w, h = img.size
                            img_left = img.crop((0, 0, h, h)).resize(
                                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                            meta_raw = tar.extractfile(
                                members[f"{prefix}.meta.json"]).read()
                            meta = json.loads(meta_raw)
                            task_key = meta.get(
                                "sub_task_instruction_key", "task_0")
                            self.samples.append({
                                "action":      action,
                                "image":       img_left,
                                "instruction": TASK_INSTRUCTIONS.get(
                                    task_key, INSTRUCTION),
                            })
                        except: pass
            except: pass

        print(f"  Loaded: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens = encode_action(s["action"])
        return {
            "action_token": tokens,
            "image":        s["image"],
            "instruction":  s["instruction"],
        }

dataset = M750Dataset(SHARDS_DIR)
sample = torch.stack([dataset[i]["action_token"] for i in range(10)])
print(f"  Token range: [{sample.min()}, {sample.max()}]")
print(f"  Sample: {sample[0].tolist()[:8]}...")

# ── LOAD MODEL — resume from v3 checkpoint ────────────────────────────────────
print(f"\n[3/5] Loading from checkpoint: {BASE_MODEL}")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel

processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True)

processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<action>"]},
    replace_additional_special_tokens=False,
)
ACTION_SPECIAL_ID = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<action>")]
print(f"  <action> token id = {ACTION_SPECIAL_ID}")
print(f"  vocab_size        = {processor.tokenizer.vocab_size}")

# Load base + existing LoRA weights
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
base.resize_token_embeddings(len(processor.tokenizer))
model = PeftModel.from_pretrained(base, BASE_MODEL, is_trainable=True).to(DEVICE)
model.train()
model.print_trainable_parameters()
print("  ✅ Resumed from v3 checkpoint")

# ── COLLATE FN ────────────────────────────────────────────────────────────────
def collate_fn(examples):
    texts  = []
    images = []

    for ex in examples:
        action_tokens    = ex["action_token"]
        action_input_ids = processor.tokenizer.vocab_size - (action_tokens + 1)
        action_input_ids[action_tokens < 0] = processor.tokenizer.pad_token_id

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": ex["instruction"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text",
                 "text": f"<|quad_start|>{'<action>' * len(action_input_ids)}<|quad_end|>"}
            ]},
        ]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([ex["image"]])

    batch = processor(
        text=texts, images=images,
        return_tensors="pt", padding=True)

    for i, ex in enumerate(examples):
        action_tokens    = ex["action_token"]
        action_input_ids = processor.tokenizer.vocab_size - (action_tokens + 1)
        action_input_ids[action_tokens < 0] = processor.tokenizer.pad_token_id
        input_ids  = batch["input_ids"][i]
        action_idx = (input_ids == ACTION_SPECIAL_ID)
        input_ids[action_idx] = action_input_ids.to(input_ids.device)
        batch["input_ids"][i] = input_ids

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    assistant_marker_id = processor.tokenizer.convert_tokens_to_ids("assistant")
    for i in range(labels.shape[0]):
        input_ids = batch["input_ids"][i].tolist()
        try:
            start_idx = input_ids.index(assistant_marker_id)
        except ValueError:
            start_idx = len(input_ids)
        labels[i, :start_idx - 1] = -100
    batch["labels"] = labels
    return batch

# ── TRAINING ──────────────────────────────────────────────────────────────────
print(f"\n[4/5] Training {MAX_STEPS} steps...")
print(f"  Resuming from v3 — expecting faster convergence")
print(f"  Watch for unique token count increasing past 15+")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, drop_last=True,
                    collate_fn=collate_fn)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_STEPS, eta_min=LR/10)

step         = 0
running_loss = 0.0
best_loss    = float("inf")
optimizer.zero_grad()

while step < MAX_STEPS:
    for batch in loader:
        if step >= MAX_STEPS:
            break
        try:
            batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(**batch)
            loss    = outputs.loss / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM
        except Exception as e:
            print(f"  Step {step} error: {e}")
            optimizer.zero_grad()
            step += 1
            continue

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        if step % 50 == 0:
            avg = running_loss / 50
            running_loss = 0.0
            print(f"  Step {step:>5}/{MAX_STEPS} | loss={avg:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            if avg < best_loss:
                best_loss = avg

        if step % SAVE_STEPS == 0:
            ckpt_dir = f"{OUTPUT_DIR}/checkpoint-{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"  💾 Saved: {ckpt_dir}")

# ── SAVE FINAL ────────────────────────────────────────────────────────────────
print("\n[5/5] Saving final checkpoint...")
final_dir = f"{OUTPUT_DIR}/checkpoint-final"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)
with open(f"{final_dir}/m750_config.json", "w") as f:
    json.dump({
        "base_model":  BASE_MODEL,
        "action_dim":  ACTION_DIM,
        "action_hz":   ACTION_HZ,
        "valid_len":   VALID_LEN,
        "vocab_size":  VOCAB_SIZE,
        "vqvae_ckpt":  VQVAE_CKPT,
        "norm_path":   NORM_PATH,
        "instruction": INSTRUCTION,
    }, f, indent=2)

print(f"\n✅ Done! Best loss: {best_loss:.4f}")
print(f"   Saved: {final_dir}")
print(f"\nTest with:")
print(f"  python /tmp/test_camera.py  (update MODEL_DIR to rdt2-m750-v4)")