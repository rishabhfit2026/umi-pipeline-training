"""
STEP 4: Fine-tune RDT2 (Qwen2.5-VL) on M750 tar shards
=========================================================
Uses pre-computed action_token.npy from shards (no VQ-VAE needed at train time)
Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python step4_finetune_rdt2.py
"""

import os, sys, glob, tarfile, io, json, torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# ── CONFIG ────────────────────────────────────────────────────────────────────
RDT2_DIR     = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
SHARDS_DIR   = "/home/rishabh/Downloads/umi-pipeline-training/shards"
VQVAE_CKPT   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-7dof/vqvae_final.pt"
OUTPUT_DIR   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v2"
# Download from HuggingFace or use local path if already cached:
BASE_MODEL   = "Qwen/Qwen2.5-VL-3B-Instruct"   # smaller = fits on single GPU
INSTRUCTION  = "pick the marker and place in the box"

ACTION_DIM   = 7
ACTION_HZ    = 24
VALID_LEN    = 27      # 27 VQ-VAE tokens per action chunk
VOCAB_SIZE   = 512     # codebook size
BATCH_SIZE   = 2
GRAD_ACCUM   = 8       # effective batch = 16
MAX_STEPS    = 3000
SAVE_STEPS   = 500
LR           = 2e-5
LORA_R       = 16
LORA_ALPHA   = 32
IMG_SIZE     = 336     # Qwen2.5-VL default
DEVICE       = "cuda:0" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, RDT2_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("  Fine-tuning RDT2 (Qwen2.5-VL) for M750 D=7")
print("="*60)
print(f"  Base model : {BASE_MODEL}")
print(f"  Device     : {DEVICE}")
print(f"  Steps      : {MAX_STEPS}")
print(f"  Batch      : {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM} effective")

# ── DATASET ───────────────────────────────────────────────────────────────────
# KEY INSIGHT: shards already have action_token.npy = pre-computed VQ-VAE tokens!
# We don't need to run VQ-VAE at training time at all.
print("\n[1/4] Loading M750 tar dataset...")

# Task instruction map (from meta.json sub_task_instruction_key)
TASK_INSTRUCTIONS = {
    "task_0": "pick the marker and place in the box",
    "task_1": "pick the marker and place in the box",
}

class M750TarDataset(Dataset):
    """
    Reads from .tar shards. Each sample has:
      - action_token.npy : (27,) int16  ← pre-computed VQ tokens ✅
      - image.jpg        : (384, 768) stereo → use left 384×384
      - meta.json        : {sub_task_instruction_key, episode, frame}
    """
    def __init__(self, shards_dir, img_size=IMG_SIZE):
        self.samples = []
        self.img_size = img_size
        tar_files = sorted(glob.glob(f"{shards_dir}/*.tar"))
        print(f"  Scanning {len(tar_files)} tar files...")

        for tar_path in tar_files:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    members = {m.name: m for m in tar.getmembers()}
                    prefixes = sorted(set(
                        n.replace(".action_token.npy", "")
                        for n in members if n.endswith(".action_token.npy")
                    ))
                    for prefix in prefixes:
                        try:
                            # Action tokens (27,) int16 — pre-computed!
                            tok_raw = tar.extractfile(
                                members[f"{prefix}.action_token.npy"]).read()
                            tokens = np.load(io.BytesIO(tok_raw))  # (27,) int16
                            if tokens.shape != (VALID_LEN,):
                                continue

                            # Image — left half of stereo
                            img_raw = tar.extractfile(
                                members[f"{prefix}.image.jpg"]).read()
                            img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                            w, h = img.size   # 768, 384
                            img_left = img.crop((0, 0, h, h))  # left 384×384
                            img_left = img_left.resize(
                                (img_size, img_size), Image.BILINEAR)

                            # Instruction from meta
                            meta_raw = tar.extractfile(
                                members[f"{prefix}.meta.json"]).read()
                            meta = json.loads(meta_raw)
                            task_key = meta.get(
                                "sub_task_instruction_key", "task_0")
                            instruction = TASK_INSTRUCTIONS.get(
                                task_key, INSTRUCTION)

                            self.samples.append({
                                "tokens":      torch.from_numpy(
                                    tokens.astype(np.int64)),   # (27,)
                                "image":       img_left,        # PIL Image
                                "instruction": instruction,
                            })
                        except: pass
            except: pass

        print(f"  Loaded: {len(self.samples)} samples")
        if not self.samples:
            raise RuntimeError("No samples loaded!")

        # Show token stats
        all_toks = torch.stack([s["tokens"] for s in self.samples[:1000]])
        print(f"  Token range: [{all_toks.min()}, {all_toks.max()}]  "
              f"(expect [0, {VOCAB_SIZE-1}])")
        zero_frac = (all_toks == 0).float().mean()
        print(f"  Zero tokens: {zero_frac*100:.1f}%  "
              f"({'⚠️ all zeros — tokens not computed' if zero_frac > 0.9 else '✅ diverse tokens'})")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

dataset = M750TarDataset(SHARDS_DIR)

# ── CHECK TOKEN QUALITY ───────────────────────────────────────────────────────
# If all tokens are zero, we need to compute them from action.npy using our VQ-VAE
all_toks = torch.stack([dataset[i]["tokens"] for i in range(min(100, len(dataset)))])
TOKENS_ARE_ZERO = (all_toks == 0).float().mean() > 0.9

if TOKENS_ARE_ZERO:
    print("\n  ⚠️  action_token.npy are all zeros — computing from VQ-VAE...")

    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29505"
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    from vqvae.models.multivqvae import MultiVQVAE
    norm_data = torch.load(
        "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt",
        map_location="cpu")
    act_mean = norm_data["mean"].to(DEVICE)
    act_std  = norm_data["std"].to(DEVICE)

    vqvae_ckpt = torch.load(VQVAE_CKPT, map_location="cpu")
    vqvae = MultiVQVAE(**vqvae_ckpt["config"]).eval().to(DEVICE)
    vqvae.load_state_dict(vqvae_ckpt["model"])

    print("  Re-tokenizing all samples with trained VQ-VAE...")
    CHUNK = 512
    for i in range(0, len(dataset), CHUNK):
        batch_actions = []
        # Load raw actions for this chunk
        tar_files = sorted(glob.glob(f"{SHARDS_DIR}/*.tar"))
        # Simpler: reload from the stored action info
        # Actually load from samples directly by reading action.npy
        pass

    # Simpler approach: reload dataset with action.npy and compute tokens
    class M750TokenizedDataset(Dataset):
        def __init__(self, shards_dir, img_size=IMG_SIZE):
            self.samples = []
            for tar_path in sorted(glob.glob(f"{shards_dir}/*.tar")):
                try:
                    with tarfile.open(tar_path, "r") as tar:
                        members = {m.name: m for m in tar.getmembers()}
                        prefixes = sorted(set(
                            n.replace(".action.npy","")
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
                                img = Image.open(io.BytesIO(img_raw)).convert("RGB")
                                w, h = img.size
                                img_left = img.crop((0, 0, h, h)).resize(
                                    (img_size, img_size), Image.BILINEAR)
                                meta_raw = tar.extractfile(
                                    members[f"{prefix}.meta.json"]).read()
                                meta = json.loads(meta_raw)
                                task_key = meta.get("sub_task_instruction_key","task_0")
                                self.samples.append({
                                    "action":      torch.from_numpy(action),
                                    "image":       img_left,
                                    "instruction": TASK_INSTRUCTIONS.get(task_key, INSTRUCTION),
                                })
                            except: pass
                except: pass
            print(f"  Reloaded: {len(self.samples)} samples with raw actions")

        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            s = self.samples[i]
            # Tokenize on the fly
            with torch.no_grad():
                act = s["action"].unsqueeze(0).to(DEVICE)
                act_norm = (act - act_mean) / act_std
                toks = vqvae.encode(act_norm).squeeze(0).cpu()
            return {"tokens": toks, "image": s["image"],
                    "instruction": s["instruction"]}

    dataset = M750TokenizedDataset(SHARDS_DIR)
    print("  ✅ Dataset re-tokenized")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=False, drop_last=True,
                    collate_fn=lambda x: x)  # return list, we handle collation

# ── LOAD QWEN2.5-VL + LoRA ───────────────────────────────────────────────────
print("\n[2/4] Loading Qwen2.5-VL + LoRA...")

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=True)

# Add <action> special token (as used by RDT2)
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<action>"]},
    replace_additional_special_tokens=False,
)
ACTION_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<action>")
print(f"  <action> token id = {ACTION_TOKEN_ID}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
# Resize embeddings for new token
model.resize_token_embeddings(len(processor.tokenizer))

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    init_lora_weights="gaussian",
)
model = get_peft_model(model, lora_config)
model = model.to(DEVICE)
model.train()
model.print_trainable_parameters()

# ── BUILD PROMPT FUNCTION ─────────────────────────────────────────────────────
def make_inputs(examples):
    """
    Build model inputs for a batch of examples.
    Each example: {"tokens": (27,), "image": PIL, "instruction": str}
    
    Format:
      <image> Task: {instruction}
      Predict the next 27 action tokens:
      <action> t1 t2 ... t27
    
    We train the model to predict the token IDs as text numbers.
    """
    messages_batch = []
    for ex in examples:
        messages_batch.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image"]},
                {"type": "text",
                 "text": f"Robot task: {ex['instruction']}\n"
                         f"Predict the next {VALID_LEN} action tokens as "
                         f"space-separated integers (0-{VOCAB_SIZE-1}):"},
            ]
        }])

    texts = [
        processor.apply_chat_template(m, tokenize=False,
                                       add_generation_prompt=True)
        for m in messages_batch
    ]

    images_batch = [[ex["image"]] for ex in examples]

    inputs = processor(
        text=texts,
        images=images_batch,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    return inputs

def make_labels(examples, input_ids):
    """
    Build labels: token sequence as text " 42 103 7 ..."
    We only supervise on the answer portion.
    """
    B = len(examples)
    labels_list = []

    for i, ex in enumerate(examples):
        # Target: space-separated token ids
        tok_str = " ".join(str(t.item()) for t in ex["tokens"])
        target_ids = processor.tokenizer.encode(
            tok_str, add_special_tokens=False)
        labels_list.append(target_ids)

    # Build label tensor: -100 for prompt, actual ids for answer
    seq_len = input_ids.shape[1]
    labels = torch.full((B, seq_len), -100, dtype=torch.long, device=DEVICE)

    for i, tgt in enumerate(labels_list):
        tgt_tensor = torch.tensor(tgt, dtype=torch.long, device=DEVICE)
        tgt_len = len(tgt_tensor)
        if tgt_len <= seq_len:
            labels[i, -tgt_len:] = tgt_tensor

    return labels

# ── TRAINING ──────────────────────────────────────────────────────────────────
print(f"\n[3/4] Training for {MAX_STEPS} steps...")
print(f"  Strategy: image + instruction → predict 27 token IDs as text")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=MAX_STEPS, eta_min=LR/10)

step = 0
running_loss = 0.0
best_loss = float("inf")
optimizer.zero_grad()

while step < MAX_STEPS:
    for batch in loader:
        if step >= MAX_STEPS:
            break

        try:
            inputs = make_inputs(batch)
            labels = make_labels(batch, inputs["input_ids"])

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

        except Exception as e:
            print(f"  Step {step} error: {e}")
            optimizer.zero_grad()
            step += 1
            continue

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        if step % 50 == 0:
            avg = running_loss / 50
            running_loss = 0.0
            print(f"  Step {step:>4}/{MAX_STEPS} | "
                  f"loss={avg:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            if avg < best_loss:
                best_loss = avg

        if step % SAVE_STEPS == 0:
            ckpt_dir = f"{OUTPUT_DIR}/checkpoint-{step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            torch.save({"step": step, "loss": avg if step%50==0 else 0},
                       f"{ckpt_dir}/train_state.pt")
            print(f"  💾 {ckpt_dir}")

# ── FINAL SAVE ────────────────────────────────────────────────────────────────
print("\n[4/4] Saving final model...")
final_dir = f"{OUTPUT_DIR}/checkpoint-final"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
processor.save_pretrained(final_dir)

# Save inference config
import json as _json
with open(f"{final_dir}/m750_config.json", "w") as f:
    _json.dump({
        "base_model":   BASE_MODEL,
        "action_dim":   ACTION_DIM,
        "action_hz":    ACTION_HZ,
        "valid_len":    VALID_LEN,
        "vocab_size":   VOCAB_SIZE,
        "vqvae_ckpt":   VQVAE_CKPT,
        "instruction":  INSTRUCTION,
        "norm_path":    "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt",
    }, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ Fine-tuning complete!  Best loss: {best_loss:.4f}")
print(f"   Saved: {final_dir}")
print(f"\nNext: python step5_inference.py")