"""
RDT2 CLIENT — runs in your umi_env terminal (NOT in Isaac Sim)
==============================================================
Run:
    cd /home/rishabh/Downloads/umi-pipeline-training/RDT2
    source /home/rishabh/Downloads/umi-pipeline-training/umi_env/bin/activate
    python rdt2_client.py

This connects to isaac_server.py running inside Isaac Sim,
receives camera frames, runs RDT2 inference, sends back actions.
"""

import sys, os, socket, struct, io, time, torch
import numpy as np
import torch.distributed as dist
from PIL import Image

RDT2_DIR   = "/home/rishabh/Downloads/umi-pipeline-training/RDT2"
MODEL_DIR  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-v3/checkpoint-final"
VQVAE_CKPT = "/home/rishabh/Downloads/umi-pipeline-training/outputs/vqvae-m750-7dof/vqvae_final.pt"
NORM_PATH  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/m750_normalizer_7dof.pt"
INSTRUCTION = "pick the marker and place in the box"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9998
VALID_LEN   = 27
VOCAB_SIZE  = 512
IMG_SIZE    = 336
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, RDT2_DIR)
print("="*50)
print("  RDT2 Inference Client")
print("="*50)
print(f"  Device: {DEVICE}")

# ── INIT ──────────────────────────────────────────────────────────────────────
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29510"
if not dist.is_initialized():
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("\n[1/3] Loading models...")
norm_data = torch.load(NORM_PATH, map_location="cpu")
act_mean  = norm_data["mean"].to(DEVICE)
act_std   = norm_data["std"].to(DEVICE)

from vqvae.models.multivqvae import MultiVQVAE
vqvae_ckpt = torch.load(VQVAE_CKPT, map_location="cpu")
vqvae = MultiVQVAE(**vqvae_ckpt["config"]).eval().to(DEVICE)
vqvae.load_state_dict(vqvae_ckpt["model"])

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

processor = AutoProcessor.from_pretrained(MODEL_DIR, use_fast=True)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager")
base.resize_token_embeddings(len(processor.tokenizer))
model = PeftModel.from_pretrained(base, MODEL_DIR).eval().to(DEVICE)

ACTION_SPECIAL_ID   = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<action>")]
VOCAB_SIZE_ACTUAL   = processor.tokenizer.vocab_size
print("  ✅ Models ready")

@torch.no_grad()
def predict(pil_image):
    img = pil_image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": [
            {"type": "text",
             "text": f"<|quad_start|>{'<action>'*VALID_LEN}<|quad_end|>"}]}
    ]
    text   = processor.apply_chat_template(messages, add_generation_prompt=False)
    inputs = processor(text=[text], images=[[img]],
                       return_tensors="pt", padding=True).to(DEVICE)

    input_ids   = inputs["input_ids"][0]
    action_mask = (input_ids == ACTION_SPECIAL_ID)
    outputs     = model(**{k: v for k, v in inputs.items() if k != "labels"})
    logits      = outputs.logits[0]

    positions = action_mask.nonzero(as_tuple=True)[0]
    token_ids = []
    for pos in positions:
        pred_id = logits[pos].argmax().item()
        tok = max(0, min(VOCAB_SIZE-1, VOCAB_SIZE_ACTUAL - (pred_id + 1)))
        token_ids.append(tok)

    while len(token_ids) < VALID_LEN:
        token_ids.append(0)
    token_ids = torch.tensor(token_ids[:VALID_LEN], dtype=torch.long)

    ids_gpu  = token_ids.unsqueeze(0).to(DEVICE)
    act_norm = vqvae.decode(ids_gpu).squeeze(0)
    act_real = (act_norm * act_std + act_mean).cpu().numpy()  # (24, 7)
    return act_real

# ── CONNECT TO ISAAC ──────────────────────────────────────────────────────────
print(f"\n[2/3] Connecting to Isaac Sim on port {SERVER_PORT}...")
print("  (Make sure isaac_server.py is running in Isaac Sim Script Editor)")

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((SERVER_HOST, SERVER_PORT))
print("  ✅ Connected!")

# ── INFERENCE LOOP ────────────────────────────────────────────────────────────
print("\n[3/3] Running inference loop...")
print("  Press Ctrl+C to stop\n")

cycle = 0
try:
    while True:
        cycle += 1

        # 1. Receive frame from Isaac Sim
        raw_len = sock.recv(4)
        if not raw_len: break
        img_len = struct.unpack(">I", raw_len)[0]

        data = b""
        while len(data) < img_len:
            chunk = sock.recv(img_len - len(data))
            if not chunk: break
            data += chunk

        img = Image.open(io.BytesIO(data)).convert("RGB")

        # 2. Run RDT2 inference
        t0 = time.time()
        actions = predict(img)   # (24, 7) real units
        dt = time.time() - t0

        print(f"  Cycle {cycle:>3} | {dt:.2f}s | "
              f"x={actions[0,0]*1000:>6.1f}mm "
              f"y={actions[0,1]*1000:>6.1f}mm "
              f"z={actions[0,2]*1000:>6.1f}mm "
              f"grip={actions[0,6]/0.06*100:>4.0f}%")

        # 3. Send actions back to Isaac Sim (168 floats)
        flat = actions.flatten().astype(np.float32)
        payload = struct.pack(f">{len(flat)}f", *flat)
        sock.sendall(struct.pack(">I", len(payload)) + payload)

except KeyboardInterrupt:
    print("\nStopped.")
finally:
    sock.close()