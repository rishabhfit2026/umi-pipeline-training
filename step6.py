"""
STEP 6: Webcam + RDT2 + SAPIEN with Real IK
=============================================
Terminal 1 (maniskill2): python step6_webcam_sim.py --mode server
Terminal 2 (umi_env):    python step6_webcam_sim.py --mode client
"""

import sys, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["server","client"], default="server")
args = parser.parse_args()

CHECKPOINT  = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-zarr/checkpoint-final"
URDF_PATH   = "/home/rishabh/Downloads/myarm_m750_fixed.urdf"
DEVICE      = "cuda:0"
WEBCAM_IDX  = 4
ACT_REPEAT  = 10
HOST        = "127.0.0.1"
PORT        = 9998
INSTRUCTION = "pick up the marker and place it in the box"

def run_server():
    import sapien
    import numpy as np, socket, pickle
    from scipy.spatial.transform import Rotation

    print("="*55)
    print("  SAPIEN Server — M750 Real IK")
    print("="*55)

    scene = sapien.Scene()
    scene.set_timestep(1/120)
    scene.set_ambient_light([0.6, 0.6, 0.6])
    scene.add_directional_light([0, 1, -1], [1, 1, 1])
    scene.add_ground(altitude=0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot  = loader.load(URDF_PATH)
    robot.set_pose(sapien.Pose(p=[0, 0, 0]))

    joints   = robot.get_active_joints()
    n_joints = len(joints)
    links    = {l.name: l for l in robot.get_links()}
    ee_link  = links['gripper']

    # Correct API from your sapien version
    ee_link_id       = ee_link.get_index()
    active_joint_ids = list(range(n_joints))
    print(f"  Joints: {n_joints}  EE link index: {ee_link_id}")

    for jt in joints:
        jt.set_drive_property(stiffness=5000, damping=500)

    q0 = np.zeros(n_joints)
    q0[1] = -0.5
    q0[2] =  0.8
    robot.set_qpos(q0)
    for i, jt in enumerate(joints):
        jt.set_drive_target(float(q0[i]))

    # Red marker
    mr = sapien.render.RenderMaterial()
    mr.base_color = [0.95, 0.1, 0.1, 1.0]
    b = scene.create_actor_builder()
    b.add_capsule_visual(radius=0.018, half_length=0.045, material=mr)
    b.add_capsule_collision(radius=0.018, half_length=0.045)
    marker = b.build(name="marker")
    marker.set_pose(sapien.Pose(p=[0.25, 0.0, 0.06]))

    # Blue box
    mb = sapien.render.RenderMaterial()
    mb.base_color = [0.1, 0.2, 0.95, 1.0]
    b2 = scene.create_actor_builder()
    b2.add_box_visual(half_size=[0.055, 0.055, 0.04], material=mb)
    b2.add_box_collision(half_size=[0.055, 0.055, 0.04])
    box = b2.build_static(name="box")
    box.set_pose(sapien.Pose(p=[0.30, 0.18, 0.04]))

    # Table
    mt = sapien.render.RenderMaterial()
    mt.base_color = [0.85, 0.85, 0.80, 1.0]
    bt = scene.create_actor_builder()
    bt.add_box_visual(half_size=[0.4, 0.4, 0.01], material=mt)
    bt.add_box_collision(half_size=[0.4, 0.4, 0.01])
    table = bt.build_static(name="table")
    table.set_pose(sapien.Pose(p=[0.25, 0.0, 0.01]))

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(0.8, -0.3, 0.7)
    viewer.set_camera_rpy(0, -0.5, 0.3)

    # ── REAL IK ──────────────────────────────────────────────────
    def solve_ik(target_pos, target_rotvec, current_qpos, n_iter=30):
        q = current_qpos.copy()
        robot.set_qpos(q)

        target_rot = Rotation.from_rotvec(target_rotvec) \
                     if np.linalg.norm(target_rotvec) > 1e-6 \
                     else Rotation.identity()

        for _ in range(n_iter):
            ee_pose = ee_link.get_pose()
            cur_pos = np.array(ee_pose.p)
            cur_rot = Rotation.from_quat([
                ee_pose.q[1], ee_pose.q[2],
                ee_pose.q[3], ee_pose.q[0]
            ])

            pos_err = target_pos - cur_pos
            if np.linalg.norm(pos_err) < 0.003:
                break

            rot_err = (target_rot * cur_rot.inv()).as_rotvec()
            twist   = np.concatenate([
                pos_err * 5.0,
                rot_err * 2.0
            ]).astype(np.float32)

            dq = robot.compute_cartesian_diff_ik(
                twist,
                ee_link_id,
                active_joint_ids
            )
            q = q + np.array(dq).flatten() * 0.3
            q = np.clip(q, -3.14, 3.14)
            robot.set_qpos(q)

        return q

    # ── SOCKET ───────────────────────────────────────────────────
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(1)
    sock.settimeout(0.01)
    print(f"  Listening on {HOST}:{PORT}")
    print("  >>> Start client in Terminal 2 <<<\n")

    conn = None; step = 0; current_q = q0.copy()

    while not viewer.closed:
        if conn is None:
            try:
                conn, addr = sock.accept()
                conn.settimeout(5.0)
                print(f"  Client connected: {addr}")
            except socket.timeout:
                pass

        if conn is not None:
            try:
                mpos = np.array(marker.get_pose().p)
                d = pickle.dumps({"n_joints": n_joints, "qpos": current_q,
                                  "step": step, "marker": mpos})
                conn.sendall(len(d).to_bytes(4, 'big') + d)

                raw = b""
                while len(raw) < 4: raw += conn.recv(4 - len(raw))
                al = int.from_bytes(raw, 'big')
                ad = b""
                while len(ad) < al: ad += conn.recv(al - len(ad))
                payload = pickle.loads(ad)

                target_pos = np.array(payload["eef_pos"])
                target_rot = np.array(payload["eef_rot"])
                gripper    = float(payload["gripper"])

                # Solve IK for arm joints only (first 6)
                arm_q     = solve_ik(target_pos, target_rot,
                                     current_q[:6], n_iter=30)
                new_q     = current_q.copy()
                new_q[:6] = arm_q[:6]

                # Gripper
                g = np.clip((gripper - 0.006) / (0.044 - 0.006), 0, 1)
                if n_joints >= 7: new_q[6] = g
                if n_joints >= 8: new_q[7] = g

                for i, jt in enumerate(joints):
                    jt.set_drive_target(float(new_q[i]))
                current_q = robot.get_qpos().copy()

                if step % 30 == 0:
                    ee_p = np.array(ee_link.get_pose().p)
                    print(f"  step={step:4d} | "
                          f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) | "
                          f"ee=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f}) | "
                          f"grip={g:.2f}")

            except Exception as e:
                print(f"  conn err: {e}"); conn = None

        for _ in range(ACT_REPEAT): scene.step()
        scene.update_render()
        viewer.render()
        step += 1

    sock.close()


def run_client():
    import torch, numpy as np, cv2, socket, pickle, time
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print("="*55)
    print("  RDT2 Client — Webcam")
    print("="*55)

    print("[1/3] Loading model...")
    processor = AutoProcessor.from_pretrained(CHECKPOINT, use_fast=True)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16, attn_implementation="eager")
    base.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base, CHECKPOINT).to(DEVICE).eval()

    cfg = torch.load(f"{CHECKPOINT}/action_config.pt", map_location="cpu")
    be  = f"{CHECKPOINT}/bin_edges.npy"
    if not os.path.exists(be):
        be = f"{os.path.dirname(CHECKPOINT)}/bin_edges.npy"
    bin_edges = np.load(be)
    AVS=cfg["act_vocab_start"]; NB=cfg["n_bins"]
    ND=cfg["n_dofs"]; PS=cfg["pred_steps"]; SL=PS*ND
    print(f"  Model ready  SEQ_LEN={SL}")

    def decode(toks):
        acts = []
        for i, v in enumerate(toks):
            d = i % ND; b = np.clip(v - AVS - d * NB, 0, NB - 1)
            acts.append((bin_edges[d][b] + bin_edges[d][b+1]) / 2)
        return np.array(acts).reshape(PS, ND)

    @torch.no_grad()
    def predict(img_pil):
        msgs = [{"role":"user","content":[{"type":"image"},
                {"type":"text","text":INSTRUCTION}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inp  = processor(text=[text], images=[[img_pil]], return_tensors="pt")
        kw   = {"input_ids":      inp["input_ids"].to(DEVICE),
                "attention_mask": inp["attention_mask"].to(DEVICE)}
        if "pixel_values" in inp:
            kw["pixel_values"]   = inp["pixel_values"].to(DEVICE, dtype=torch.bfloat16)
            kw["image_grid_thw"] = inp["image_grid_thw"].to(DEVICE)
        gen = []; pkv = None
        for si in range(SL):
            out = model(**kw, use_cache=True, past_key_values=pkv)
            pkv = out.past_key_values
            d = si % ND; s = AVS + d * NB
            nt = s + out.logits[0, -1, s:s+NB].argmax().item()
            gen.append(nt)
            kw = {"input_ids":      torch.tensor([[nt]], device=DEVICE),
                  "attention_mask": torch.ones((1,1), device=DEVICE)}
        return decode(gen)

    print(f"[2/3] Opening webcam {WEBCAM_IDX}...")
    cap = cv2.VideoCapture(WEBCAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"  Cannot open webcam {WEBCAM_IDX}"); return
    for _ in range(30): cap.read()
    print("  Webcam ready")

    print(f"[3/3] Connecting to {HOST}:{PORT}...")
    sock = socket.socket()
    while True:
        try: sock.connect((HOST, PORT)); break
        except: print("  waiting..."); time.sleep(1)
    print("  Connected!\n")
    print("  Point webcam at: white mat + red marker + box")
    print("  Press Q to quit\n")

    queue = []; nc = 0; fn = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fn += 1

        disp = frame.copy()
        cv2.putText(disp, f"RDT2 Webcam | inf#{nc}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if queue:
            a = queue[0]
            cv2.putText(disp,
                f"EEF:({a[0]:.3f},{a[1]:.3f},{a[2]:.3f}) grip:{a[6]:.3f}",
                (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
        cv2.imshow("Webcam Input", disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        try:
            raw = b""
            while len(raw) < 4: raw += sock.recv(4 - len(raw))
            dl = int.from_bytes(raw, 'big'); d = b""
            while len(d) < dl: d += sock.recv(dl - len(d))
            state = pickle.loads(d); step = state["step"]

            if not queue:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((224, 224))
                t0  = time.time()
                acts = predict(img); queue = list(acts); nc += 1
                a0 = acts[0]
                print(f"  step={step:4d} inf#{nc:3d} | "
                      f"eef=({a0[0]:.3f},{a0[1]:.3f},{a0[2]:.3f}) | "
                      f"rot=({a0[3]:.2f},{a0[4]:.2f},{a0[5]:.2f}) | "
                      f"grip={a0[6]:.3f} | t={time.time()-t0:.1f}s")

            act     = queue.pop(0)
            payload = {"eef_pos": act[:3], "eef_rot": act[3:6], "gripper": act[6]}
            rep     = pickle.dumps(payload)
            sock.sendall(len(rep).to_bytes(4, 'big') + rep)

        except Exception as e:
            print(f"  err: {e}"); break

    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    print(f"  Done — {fn} frames  {nc} inferences")


if args.mode == "server": run_server()
else: run_client()
