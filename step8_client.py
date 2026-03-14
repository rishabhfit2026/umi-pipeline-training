"""
Step8 Client — RDT2 inference
source umi_env/bin/activate
python step8_client.py
"""
import torch, numpy as np, cv2, socket, pickle, time, os
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

CHECKPOINT = '/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-sim2/checkpoint-final'
DEVICE     = 'cuda:0'
HOST,PORT  = '127.0.0.1', 9997
INSTRUCTION= 'pick up the marker and place it in the box'

print("="*55)
print("  RDT2 Client — sim2 checkpoint")
print("="*55)
print("Loading model...")

processor = AutoProcessor.from_pretrained(CHECKPOINT, use_fast=True)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="eager")
base.resize_token_embeddings(len(processor.tokenizer))
model = PeftModel.from_pretrained(base, CHECKPOINT).to(DEVICE).eval()

cfg       = torch.load(f"{CHECKPOINT}/action_config.pt", map_location="cpu")
bin_edges = np.load(f"{CHECKPOINT}/bin_edges.npy")
AVS=cfg["act_vocab_start"]; NB=cfg["n_bins"]
ND=cfg["n_dofs"]; PS=cfg["pred_steps"]; SL=PS*ND
print(f"Model ready  SEQ_LEN={SL}  AVS={AVS}")

def decode(toks):
    acts=[]
    for i,v in enumerate(toks):
        d=i%ND; b=np.clip(v-AVS-d*NB,0,NB-1)
        acts.append((bin_edges[d][b]+bin_edges[d][b+1])/2)
    return np.array(acts).reshape(PS,ND)

@torch.no_grad()
def predict(img_pil):
    msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":INSTRUCTION}]}]
    text=processor.apply_chat_template(msgs,add_generation_prompt=True)
    inp=processor(text=[text],images=[[img_pil]],return_tensors="pt")
    kw={"input_ids":inp["input_ids"].to(DEVICE),
        "attention_mask":inp["attention_mask"].to(DEVICE)}
    if "pixel_values" in inp:
        kw["pixel_values"]=inp["pixel_values"].to(DEVICE,dtype=torch.bfloat16)
        kw["image_grid_thw"]=inp["image_grid_thw"].to(DEVICE)
    gen=[]; pkv=None
    for si in range(SL):
        out=model(**kw,use_cache=True,past_key_values=pkv)
        pkv=out.past_key_values
        d=si%ND; s=AVS+d*NB
        nt=s+out.logits[0,-1,s:s+NB].argmax().item()
        gen.append(nt)
        kw={"input_ids":torch.tensor([[nt]],device=DEVICE),
            "attention_mask":torch.ones((1,1),device=DEVICE)}
    return decode(gen)

print(f"Connecting to {HOST}:{PORT}...")
sock=socket.socket()
while True:
    try: sock.connect((HOST,PORT)); break
    except: print("  waiting..."); time.sleep(1)
print("Connected!\n")

queue=[]; nc=0; fn=0
while True:
    try:
        raw=b""
        while len(raw)<4: raw+=sock.recv(4-len(raw))
        dl=int.from_bytes(raw,'big'); d=b""
        while len(d)<dl: d+=sock.recv(dl-len(d))
        state  = pickle.loads(d)
        img_np = state["img"]
        step   = state["step"]
        mpos   = state["marker_pos"]
        ee_pos = state["ee_pos"]

        # Show what model sees
        bgr = img_np[:,:,::-1].copy()
        cv2.putText(bgr,f"step={step} inf={nc}",(3,15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
        cv2.putText(bgr,f"ee=({ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f})",(3,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,255),1)
        cv2.imshow("Sim Camera — RDT2 Input", bgr)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        fn+=1

        if not queue:
            t0=time.time()
            acts=predict(Image.fromarray(img_np))
            queue=list(acts); nc+=1
            a0=acts[0]
            print(f"  step={step:4d} inf#{nc:3d} | "
                  f"eef=({a0[0]:.3f},{a0[1]:.3f},{a0[2]:.3f}) | "
                  f"grip={a0[6]:.4f} | t={time.time()-t0:.1f}s")

        act=queue.pop(0)
        payload={"eef_pos":act[:3].tolist(),"eef_rot":act[3:6].tolist(),"gripper":float(act[6])}
        rep=pickle.dumps(payload)
        sock.sendall(len(rep).to_bytes(4,'big')+rep)

    except Exception as e:
        import traceback; traceback.print_exc(); break

cv2.destroyAllWindows(); sock.close()