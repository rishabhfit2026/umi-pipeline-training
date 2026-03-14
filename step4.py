import sys, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["server","client"], default="server")
args = parser.parse_args()

CHECKPOINT  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-zarr/checkpoint-final"
URDF_PATH   = "/home/rishabh/Downloads/myarm_m750_fixed.urdf"
DEVICE      = "cuda:0"
ACT_REPEAT  = 15
IMG_SIZE    = 224
HOST        = "127.0.0.1"
PORT        = 9999
INSTRUCTION = "pick up the marker and place it in the box"

def run_server():
    import sapien, numpy as np, socket, pickle
    print("SAPIEN Server - M750 Pick and Place")
    scene = sapien.Scene()
    scene.set_timestep(1/60)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1])
    scene.add_ground(0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(URDF_PATH)
    robot.set_pose(sapien.Pose(p=[0, 0, 0]))
    joints = robot.get_active_joints()
    n_joints = len(joints)
    print(f"  Joints: {n_joints}")

    for jt in joints:
        jt.set_drive_property(stiffness=2000, damping=200)
    qpos0 = np.zeros(n_joints)
    robot.set_qpos(qpos0)
    for i, jt in enumerate(joints):
        jt.set_drive_target(float(qpos0[i]))

    # Red marker
    mr = sapien.render.RenderMaterial()
    mr.base_color = [0.9, 0.1, 0.1, 1.0]
    b = scene.create_actor_builder()
    b.add_capsule_visual(radius=0.015, half_length=0.04, material=mr)
    b.add_capsule_collision(radius=0.015, half_length=0.04)
    marker = b.build(name="marker")
    marker.set_pose(sapien.Pose(p=[0.25, 0.0, 0.06]))

    # Blue box
    mb = sapien.render.RenderMaterial()
    mb.base_color = [0.1, 0.2, 0.9, 1.0]
    b2 = scene.create_actor_builder()
    b2.add_box_visual(half_size=[0.05, 0.05, 0.03], material=mb)
    b2.add_box_collision(half_size=[0.05, 0.05, 0.03])
    box = b2.build_static(name="box")
    box.set_pose(sapien.Pose(p=[0.30, 0.18, 0.03]))

    # Camera
    ce = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(IMG_SIZE, IMG_SIZE)
    cam.set_fovy(0.9)
    cam.near = 0.01
    cam.far = 10.0
    ce.add_component(cam)
    ce.set_pose(sapien.Pose(p=[0.6, 0.0, 0.7], q=[0.707, 0, 0.707, 0]))
    scene.add_entity(ce)

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(0.8, 0.0, 0.6)
    viewer.set_camera_rpy(0, -0.4, 0)

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)
    sock.settimeout(0.01)
    print(f"  Listening on {HOST}:{PORT}")
    print("  >>> Now run:  python step4.py --mode client  in another terminal <<<")

    conn = None
    step = 0
    while not viewer.closed:
        if conn is None:
            try:
                conn, addr = sock.accept()
                conn.settimeout(5.0)
                print(f"  Client connected: {addr}")
            except socket.timeout:
                pass

        scene.update_render()
        cam.take_picture()
        rgba = cam.get_picture("Color")
        img = (rgba[:, :, :3] * 255).clip(0, 255).astype("uint8")

        if conn is not None:
            try:
                d = pickle.dumps({"img": img, "n_joints": n_joints, "step": step})
                conn.sendall(len(d).to_bytes(4, 'big') + d)
                raw = b""
                while len(raw) < 4:
                    raw += conn.recv(4 - len(raw))
                al = int.from_bytes(raw, 'big')
                ad = b""
                while len(ad) < al:
                    ad += conn.recv(al - len(ad))
                tgts = pickle.loads(ad)
                for i, jt in enumerate(joints):
                    jt.set_drive_target(float(tgts[i]))
                if step % 50 == 0:
                    print(f"  step={step}  qpos={robot.get_qpos()[:3].round(3)}")
            except Exception as e:
                print(f"  conn err: {e}")
                conn = None

        for _ in range(ACT_REPEAT):
            scene.step()
        scene.update_render()
        viewer.render()
        step += 1

    sock.close()


def run_client():
    import torch, numpy as np, socket, pickle, time
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print("RDT2 Client - loading model...")
    processor = AutoProcessor.from_pretrained(CHECKPOINT, use_fast=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    base.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base, CHECKPOINT).to(DEVICE).eval()

    cfg = torch.load(f"{CHECKPOINT}/action_config.pt", map_location="cpu")
    be = f"{CHECKPOINT}/bin_edges.npy"
    if not os.path.exists(be):
        be = f"{os.path.dirname(CHECKPOINT)}/bin_edges.npy"
    bin_edges = np.load(be)

    AVS = cfg["act_vocab_start"]
    NB  = cfg["n_bins"]
    ND  = cfg["n_dofs"]
    PS  = cfg["pred_steps"]
    SL  = PS * ND
    print(f"  Model ready  SEQ_LEN={SL}")

    def decode(toks):
        acts = []
        for i, v in enumerate(toks):
            d = i % ND
            b = np.clip(v - AVS - d * NB, 0, NB - 1)
            acts.append((bin_edges[d][b] + bin_edges[d][b+1]) / 2)
        return np.array(acts).reshape(PS, ND)

    @torch.no_grad()
    def predict(img_pil):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp = processor(text=[text], images=[[img_pil]], return_tensors="pt")
        kw = {"input_ids": inp["input_ids"].to(DEVICE), "attention_mask": inp["attention_mask"].to(DEVICE)}
        if "pixel_values" in inp:
            kw["pixel_values"] = inp["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
            kw["image_grid_thw"] = inp["image_grid_thw"].to(DEVICE)
        gen = []; pkv = None
        for si in range(SL):
            out = model(**kw, use_cache=True, past_key_values=pkv)
            pkv = out.past_key_values
            d = si % ND
            s = AVS + d * NB
            nt = s + out.logits[0, -1, s:s+NB].argmax().item()
            gen.append(nt)
            kw = {"input_ids": torch.tensor([[nt]], device=DEVICE),
                  "attention_mask": torch.ones((1, 1), device=DEVICE)}
        return decode(gen)

    def to_qpos(a7, nj):
        q = np.zeros(nj)
        if nj >= 1: q[0] = np.arctan2(a7[1], a7[0])
        if nj >= 2: q[1] = np.clip(np.arctan2(a7[2], np.sqrt(a7[0]**2 + a7[1]**2)) - 0.3, -1.5, 1.5)
        if nj >= 3: q[2] = np.clip(-q[1] * 0.8, -1.5, 1.5)
        if nj >= 4: q[3] = np.clip(a7[3] * 0.5, -1.5, 1.5)
        if nj >= 5: q[4] = np.clip(a7[4] * 0.5, -1.5, 1.5)
        if nj >= 6: q[5] = np.clip(a7[5] * 0.5, -1.5, 1.5)
        g = (a7[6] - 0.006) / (0.044 - 0.006)
        if nj >= 7: q[6] = g
        if nj >= 8: q[7] = g
        return q

    sock = socket.socket()
    print(f"  Connecting to {HOST}:{PORT}...")
    while True:
        try:
            sock.connect((HOST, PORT))
            break
        except ConnectionRefusedError:
            print("  waiting for server...")
            time.sleep(1)
    print("  Connected!")

    queue = []; nc = 0
    while True:
        try:
            r = b""
            while len(r) < 4:
                r += sock.recv(4 - len(r))
            dl = int.from_bytes(r, 'big')
            d = b""
            while len(d) < dl:
                d += sock.recv(dl - len(d))
            p = pickle.loads(d)
            img = p["img"]; nj = p["n_joints"]; st = p["step"]

            if not queue:
                t0 = time.time()
                acts = predict(Image.fromarray(img))
                queue = list(acts)
                nc += 1
                print(f"  step={st} inf#{nc} eef={acts[0,:3].round(3)} grip={acts[0,6]:.3f} t={time.time()-t0:.1f}s")

            act = queue.pop(0)
            tgt = to_qpos(act, nj)
            rep = pickle.dumps(tgt)
            sock.sendall(len(rep).to_bytes(4, 'big') + rep)
        except Exception as e:
            print(f"  err: {e}")
            break
    sock.close()


if args.mode == "server":
    run_server()
else:
    run_client()