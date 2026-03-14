"""
STEP 8 (sim2 update): ArUco + RDT2 + SAPIEN
Now uses rdt2-m750-sim2/checkpoint-final trained on sim data
"""

import sys, os, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["server","client","print_markers"], default="server")
args = parser.parse_args()

# ── ONLY THIS CHANGED ────────────────────────────────────────────
CHECKPOINT   = "/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-sim2/checkpoint-final"
# ─────────────────────────────────────────────────────────────────

URDF_PATH    = "/home/rishabh/Downloads/myarm_m750_fixed.urdf"
DEVICE       = "cuda:0"
WEBCAM_IDX   = 4
HOST         = "127.0.0.1"
PORT         = 9996
INSTRUCTION  = "pick up the marker and place it in the box"
IMG_SIZE     = 224
ACT_REPEAT   = 10

ARUCO_DICT   = cv2 = None
MARKER_SIZE  = 0.04
CAMERA_H     = 0.75
MARKER_ID    = 0
BOX_ID       = 1

def print_markers():
    import cv2
    import numpy as np
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    os.makedirs("aruco_markers", exist_ok=True)
    for mid in [0, 1]:
        img = cv2.aruco.generateImageMarker(aruco_dict, mid, 300)
        bordered = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        fname = f"aruco_markers/marker_id{mid}.png"
        cv2.imwrite(fname, bordered)
        print(f"  Saved: {fname}  (print at 4x4cm)")
    print("\n  Instructions:")
    print("  1. Print marker_id0.png — stick on RED marker object")
    print("  2. Print marker_id1.png — stick on BOX")
    print("  3. Mount webcam 70-80cm above table looking DOWN")
    print("  4. Run server + client")

def run_server():
    import sapien
    import cv2
    import numpy as np, socket, pickle
    from scipy.spatial.transform import Rotation

    print("="*60)
    print("  SAPIEN Server — sim2 model + ArUco + Pinocchio IK")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("="*60)

    # Auto-detect webcam
    cap = None
    for idx in [WEBCAM_IDX, 0, 1, 2, 3, 4]:
        test = cv2.VideoCapture(idx)
        ret, _ = test.read()
        if ret:
            cap = test
            print(f"  Webcam found at index {idx}")
            break
        test.release()
    if cap is None:
        print("  WARNING: No webcam — sim-only mode")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(30): cap.read()

    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    fx = fy = 600.0; cx, cy = 320.0, 240.0
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    D = np.zeros(5, dtype=np.float32)

    # Default positions matching sim training data (TX=0.253, TY=0.0)
    last_marker_pos = np.array([0.251, 0.0, 0.095])
    last_box_pos    = np.array([0.251, 0.12, 0.095])

    def detect_objects(frame):
        nonlocal last_marker_pos, last_box_pos
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        marker_pos = last_marker_pos.copy()
        box_pos    = last_box_pos.copy()
        detected   = []
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                obj_pts = np.array([
                    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
                    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
                ], dtype=np.float32)
                _, rvec, tvec = cv2.solvePnP(obj_pts, corners[i][0], K, D)
                tvec = tvec.flatten()
                real_x = np.clip( tvec[2]*0.85,        0.15, 0.40)
                real_y = np.clip(-tvec[0]*0.85,        -0.20, 0.20)
                real_z = 0.076
                pos = np.array([real_x, real_y, real_z])
                detected.append(mid)
                if mid == MARKER_ID:
                    marker_pos = pos; last_marker_pos = pos
                elif mid == BOX_ID:
                    box_pos = pos;    last_box_pos = pos
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return marker_pos, box_pos, detected, frame

    # SAPIEN scene
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
    ee_idx   = ee_link.get_index()
    pm       = robot.create_pinocchio_model()

    for jt in joints: jt.set_drive_property(stiffness=5000, damping=500)
    q0 = np.zeros(n_joints); q0[1]=-0.3; q0[2]=0.5
    robot.set_qpos(q0)
    for i, jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(200): scene.step()

    # Get real EE home (same as training)
    for _ in range(200): scene.step()
    real_ee = np.array(ee_link.get_entity_pose().p)
    TX, TY  = real_ee[0], real_ee[1]
    print(f"  EE home: ({TX:.3f}, {TY:.3f}, {real_ee[2]:.3f})")

    # Marker (no collision — passes through)
    mr = sapien.render.RenderMaterial(); mr.base_color=[0.95,0.1,0.1,1.0]
    bm = scene.create_actor_builder()
    bm.add_capsule_visual(radius=0.018, half_length=0.050, material=mr)
    sim_marker = bm.build(name="marker")
    sim_marker.set_pose(sapien.Pose(p=last_marker_pos.tolist()))

    # Green box
    mg = sapien.render.RenderMaterial(); mg.base_color=[0.1,0.85,0.2,1.0]
    bg = scene.create_actor_builder()
    bg.add_box_visual(half_size=[0.05,0.05,0.022], material=mg)
    bg.add_box_collision(half_size=[0.05,0.05,0.022])
    sim_box = bg.build_static(name="box")
    sim_box.set_pose(sapien.Pose(p=[last_box_pos[0],last_box_pos[1],0.050]))

    # Table — same as training
    def sbox(half, col, pos, name=""):
        mt = sapien.render.RenderMaterial(); mt.base_color=col
        b  = scene.create_actor_builder()
        b.add_box_visual(half_size=half, material=mt)
        b.add_box_collision(half_size=half)
        a  = b.build_static(name=name)
        a.set_pose(sapien.Pose(p=pos)); return a

    sbox([0.30,0.28,0.025], [0.52,0.33,0.15,1.0], [TX,TY,0.050],   "table")
    for lx,ly in [(TX+0.27,TY+0.23),(TX+0.27,TY-0.23),(TX-0.27,TY+0.23),(TX-0.27,TY-0.23)]:
        sbox([0.02,0.02,0.025], [0.38,0.22,0.10,1.0], [lx,ly,0.012], "leg")
    sbox([0.20,0.18,0.002],  [0.96,0.96,0.94,1.0], [TX,TY,0.077],  "mat")

    # Virtual camera — EXACTLY same as training (step10)
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(IMG_SIZE, IMG_SIZE)
    cam.set_fovy(float(np.deg2rad(60)), True)
    cam_entity.add_component(cam)
    cam_entity.set_pose(sapien.Pose(
        p=[TX, TY, 0.85],
        q=[0, 0.7071068, 0.7071068, 0]   # straight down — matches step10
    ))
    scene.add_entity(cam_entity)

    viewer = scene.create_viewer()
    viewer.set_camera_xyz(0.8, -0.8, 1.2)
    viewer.set_camera_rpy(0, -0.55, 0.65)

    def solve_ik(target_pos, gripper_val, current_q):
        rot   = Rotation.from_euler('xyz', [np.pi, 0, 0]); qxyzw = rot.as_quat()
        tpose = sapien.Pose(p=target_pos.tolist(),
                            q=[qxyzw[3],qxyzw[0],qxyzw[1],qxyzw[2]])
        mask  = np.ones(n_joints, dtype=np.int32); mask[6:]=0
        qres, ok, _ = pm.compute_inverse_kinematics(
            ee_idx, tpose,
            initial_qpos=current_q.astype(np.float64),
            active_qmask=mask, max_iterations=300)
        q = np.array(qres)
        if n_joints>=7: q[6]=gripper_val
        if n_joints>=8: q[7]=gripper_val
        return q, ok

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT)); sock.listen(1); sock.settimeout(0.01)
    print(f"  Listening on {HOST}:{PORT}")
    print("  >>> Start client in Terminal 2 <<<\n")

    conn=None; step=0; current_q=q0.copy()
    marker_grasped=False

    while not viewer.closed:
        ret, frame = cap.read() if cap else (False, None)
        if ret:
            mpos, bpos, detected, frame = detect_objects(frame)
            if not marker_grasped:
                sim_marker.set_pose(sapien.Pose(p=mpos.tolist()))
            if BOX_ID in detected:
                sim_box.set_pose(sapien.Pose(p=[bpos[0],bpos[1],0.050]))
            cv2.putText(frame, f"marker=({mpos[0]:.2f},{mpos[1]:.2f}) det:{detected}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("Webcam ArUco", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if marker_grasped:
            p = np.array(ee_link.get_pose().p)
            sim_marker.set_pose(sapien.Pose(p=p.tolist()))

        if conn is None:
            try:
                conn, addr = sock.accept()
                conn.settimeout(5.0)
                print(f"  Client connected: {addr}")
            except socket.timeout:
                pass

        if conn is not None:
            try:
                scene.update_render(); cam.take_picture()
                rgba = cam.get_picture("Color")
                img  = (rgba[:,:,:3]*255).clip(0,255).astype("uint8")
                mpos_now = np.array(sim_marker.get_pose().p)
                ee_p     = np.array(ee_link.get_pose().p)

                d = pickle.dumps({"img": img, "n_joints": n_joints,
                    "qpos": current_q, "step": step,
                    "marker_pos": mpos_now, "ee_pos": ee_p,
                    "marker_grasped": marker_grasped})
                conn.sendall(len(d).to_bytes(4,'big')+d)

                raw=b""
                while len(raw)<4: raw+=conn.recv(4-len(raw))
                al=int.from_bytes(raw,'big'); ad=b""
                while len(ad)<al: ad+=conn.recv(al-len(ad))
                payload = pickle.loads(ad)

                target_pos   = np.array(payload["eef_pos"])
                gripper_val  = float(payload["gripper"])
                g_norm       = np.clip(gripper_val / 0.0345, 0, 1)

                new_q, ok = solve_ik(target_pos, g_norm*0.0345, current_q)
                for i,jt in enumerate(joints): jt.set_drive_target(float(new_q[i]))
                current_q = robot.get_qpos().copy()

                # Grasp detection
                dist = np.linalg.norm(ee_p - mpos_now)
                if not marker_grasped and gripper_val < 0.005 and dist < 0.12:
                    marker_grasped = True
                    print(f"  ✅ GRASPED! dist={dist:.3f}")
                if marker_grasped and gripper_val > 0.02:
                    marker_grasped = False
                    mp = np.array(sim_marker.get_pose().p)
                    bp = np.array(sim_box.get_pose().p)
                    if np.linalg.norm(mp[:2]-bp[:2]) < 0.12:
                        print(f"  🎉 PLACED in box!")
                    else:
                        print(f"  📍 Released at {np.round(mp,3)}")

                if step%30==0:
                    print(f"  step={step:4d} | IK={'✅' if ok else '❌'} | "
                          f"ee=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f}) | "
                          f"grip={gripper_val:.4f} | grasped={marker_grasped}")
            except Exception as e:
                print(f"  conn err: {e}"); conn=None

        for _ in range(ACT_REPEAT): scene.step()
        scene.update_render(); viewer.render()
        step+=1

    if cap: cap.release()
    cv2.destroyAllWindows(); sock.close()


def run_client():
    import torch, numpy as np, cv2, socket, pickle, time
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print("="*60)
    print("  RDT2 Client — sim2 checkpoint")
    print(f"  Checkpoint: {CHECKPOINT}")
    print("="*60)

    print("[1/2] Loading model...")
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
    print(f"  Model ready  SEQ_LEN={SL}  act_vocab_start={AVS}")

    def decode(toks):
        acts=[]
        for i,v in enumerate(toks):
            d=i%ND; b=np.clip(v-AVS-d*NB,0,NB-1)
            acts.append((bin_edges[d][b]+bin_edges[d][b+1])/2)
        return np.array(acts).reshape(PS,ND)

    @torch.no_grad()
    def predict(img_pil):
        msgs=[{"role":"user","content":[{"type":"image"},
               {"type":"text","text":INSTRUCTION}]}]
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

    print(f"[2/2] Connecting to {HOST}:{PORT}...")
    sock=socket.socket()
    while True:
        try: sock.connect((HOST,PORT)); break
        except: print("  waiting..."); time.sleep(1)
    print("  Connected!\n")

    queue=[]; nc=0; fn=0
    while True:
        try:
            raw=b""
            while len(raw)<4: raw+=sock.recv(4-len(raw))
            dl=int.from_bytes(raw,'big'); d=b""
            while len(d)<dl: d+=sock.recv(dl-len(d))
            state   = pickle.loads(d)
            img_np  = state["img"]
            step    = state["step"]
            mpos    = state["marker_pos"]
            ee_pos  = state["ee_pos"]
            grasped = state["marker_grasped"]

            bgr = img_np[:,:,::-1].copy()
            cv2.putText(bgr, f"sim2 model | inf#{nc} step#{step}",
                        (3,15), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
            cv2.putText(bgr, f"marker:({mpos[0]:.2f},{mpos[1]:.2f})",
                        (3,30), cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,200,255),1)
            cv2.putText(bgr, f"grasped:{grasped}",
                        (3,45), cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,100,0),1)
            cv2.imshow("Sim Camera — RDT2 sim2", bgr)
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
            print(f"  err: {e}"); import traceback; traceback.print_exc(); break

    cv2.destroyAllWindows(); sock.close()
    print(f"  Done — {fn} frames  {nc} inferences")


if args.mode=="print_markers":   print_markers()
elif args.mode=="server":        run_server()
else:                            run_client()