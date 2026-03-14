"""
STEP 7: Fully Autonomous Simulation
=====================================
Virtual camera → RDT2 model → Pinocchio IK → M750 picks marker → places in box
Everything inside SAPIEN. No webcam needed.

Terminal 1 (maniskill2): python step7_autonomous_sim.py --mode server
Terminal 2 (umi_env):    python step7_autonomous_sim.py --mode client
"""

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
PORT        = 9997
INSTRUCTION = "pick up the marker and place it in the box"

# ══════════════════════════════════════════════════════════════════
#  SERVER — full sim with virtual camera + pinocchio IK + grasping
# ══════════════════════════════════════════════════════════════════
def run_server():
    import sapien
    import numpy as np, socket, pickle
    from scipy.spatial.transform import Rotation

    print("="*60)
    print("  SAPIEN Autonomous Sim — Virtual Cam + Pinocchio IK")
    print("="*60)

    scene = sapien.Scene()
    scene.set_timestep(1/120)
    scene.set_ambient_light([0.6, 0.6, 0.6])
    scene.add_directional_light([0,  1, -1], [1, 1, 1])
    scene.add_directional_light([1, -1, -1], [0.4, 0.4, 0.4])
    scene.add_ground(altitude=0)

    # Robot
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot  = loader.load(URDF_PATH)
    robot.set_pose(sapien.Pose(p=[0, 0, 0]))

    joints   = robot.get_active_joints()
    n_joints = len(joints)
    links    = {l.name: l for l in robot.get_links()}
    ee_link  = links['gripper']
    ee_idx   = ee_link.get_index()
    print(f"  Robot: {n_joints} joints  EE link index: {ee_idx}")

    # Pinocchio IK model
    pm = robot.create_pinocchio_model()
    # Set link/joint order for pinocchio
    link_names  = [l.name for l in robot.get_links()]
    joint_names = [j.name for j in robot.get_active_joints()]
    print(f"  Pinocchio IK ready — tested OK")

    # Drive properties
    for jt in joints:
        jt.set_drive_property(stiffness=5000, damping=500)

    # Home pose
    q0 = np.zeros(n_joints)
    q0[1] = -0.5; q0[2] = 0.8
    robot.set_qpos(q0)
    for i, jt in enumerate(joints):
        jt.set_drive_target(float(q0[i]))

    # ── OBJECTS ──────────────────────────────────────────────────
    # Red marker
    mr = sapien.render.RenderMaterial()
    mr.base_color = [0.95, 0.1, 0.1, 1.0]
    bm = scene.create_actor_builder()
    bm.add_capsule_visual(radius=0.018, half_length=0.045, material=mr)
    bm.add_capsule_collision(radius=0.018, half_length=0.045)
    marker = bm.build(name="marker")
    marker.set_pose(sapien.Pose(p=[0.25, 0.0, 0.065]))

    # Green box (target)
    mg = sapien.render.RenderMaterial()
    mg.base_color = [0.1, 0.85, 0.2, 1.0]
    bg = scene.create_actor_builder()
    bg.add_box_visual(half_size=[0.06, 0.06, 0.04], material=mg)
    bg.add_box_collision(half_size=[0.06, 0.06, 0.04])
    box = bg.build_static(name="box")
    box.set_pose(sapien.Pose(p=[0.30, 0.20, 0.04]))

    # Table
    mt = sapien.render.RenderMaterial()
    mt.base_color = [0.9, 0.9, 0.85, 1.0]
    bt = scene.create_actor_builder()
    bt.add_box_visual(half_size=[0.45, 0.45, 0.01], material=mt)
    bt.add_box_collision(half_size=[0.45, 0.45, 0.01])
    table = bt.build_static(name="table")
    table.set_pose(sapien.Pose(p=[0.25, 0.0, 0.01]))

    # ── VIRTUAL CAMERA (matches training: top-down view) ─────────
    cam_entity = sapien.Entity()
    cam = sapien.render.RenderCameraComponent(IMG_SIZE, IMG_SIZE)
    cam.set_fovy(0.85)   # ~49 degrees — similar to fisheye training cam
    cam.near = 0.01; cam.far = 10.0
    cam_entity.add_component(cam)
    # Position: 70cm above workspace, looking straight down
    # Matches your training camera position
    cam_entity.set_pose(sapien.Pose(
        p=[0.25, 0.05, 0.75],
        q=[0.707, 0.0, 0.707, 0.0]   # looking down
    ))
    scene.add_entity(cam_entity)

    # Viewer (human view — different angle from virtual cam)
    viewer = scene.create_viewer()
    viewer.set_camera_xyz(0.9, -0.4, 0.8)
    viewer.set_camera_rpy(0, -0.5, 0.4)

    # ── PINOCCHIO IK SOLVER ──────────────────────────────────────
    def solve_ik_pinocchio(target_pos, target_rotvec, current_qpos):
        """
        Full IK using pinocchio CLIK algorithm.
        Returns (qpos, success)
        """
        # Build target pose in robot base frame
        if np.linalg.norm(target_rotvec) > 1e-6:
            rot = Rotation.from_rotvec(target_rotvec)
        else:
            # Default: EEF pointing down
            rot = Rotation.from_euler('xyz', [np.pi, 0, 0])

        quat_xyzw = rot.as_quat()  # [x,y,z,w]
        # SAPIEN pose uses [w,x,y,z]
        target_pose = sapien.Pose(
            p=target_pos.tolist(),
            q=[quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )

        # active_qmask: 1 for arm joints (0-5), 0 for gripper (6-7)
        mask = np.ones(n_joints, dtype=np.int32)
        mask[6:] = 0   # don't move gripper via IK

        qpos_result, success, error = pm.compute_inverse_kinematics(
            link_index     = ee_idx,
            pose           = target_pose,
            initial_qpos   = current_qpos.astype(np.float64),
            active_qmask   = mask,
            max_iterations = 200,
            dt             = 0.1,
            damp           = 1e-6
        )
        return np.array(qpos_result), success

    # ── GRASP LOGIC ──────────────────────────────────────────────
    marker_grasped    = False
    grasp_offset      = None   # marker offset relative to EE when grasped

    def try_grasp(gripper_val):
        """If gripper closes and marker is close to EE, attach marker."""
        nonlocal marker_grasped, grasp_offset
        if marker_grasped:
            # Move marker with EE
            ee_pose     = ee_link.get_pose()
            ee_pos      = np.array(ee_pose.p)
            ee_rot      = Rotation.from_quat([ee_pose.q[1], ee_pose.q[2],
                                              ee_pose.q[3], ee_pose.q[0]])
            new_mpos    = ee_pos + ee_rot.apply(grasp_offset)
            marker.set_pose(sapien.Pose(p=new_mpos.tolist()))
            return

        if gripper_val < 0.3:   # gripper closing
            ee_pose  = ee_link.get_pose()
            ee_pos   = np.array(ee_pose.p)
            mpos     = np.array(marker.get_pose().p)
            dist     = np.linalg.norm(ee_pos - mpos)
            if dist < 0.08:     # close enough to grasp
                ee_rot      = Rotation.from_quat([ee_pose.q[1], ee_pose.q[2],
                                                  ee_pose.q[3], ee_pose.q[0]])
                grasp_offset = ee_rot.inv().apply(mpos - ee_pos)
                marker_grasped = True
                print(f"  ✅ GRASPED marker! dist={dist:.3f}")

    def try_release(gripper_val):
        """If gripper opens, release marker."""
        nonlocal marker_grasped, grasp_offset
        if marker_grasped and gripper_val > 0.6:
            marker_grasped = False
            grasp_offset   = None
            mpos = np.array(marker.get_pose().p)
            bpos = np.array(box.get_pose().p)
            dist = np.linalg.norm(mpos[:2] - bpos[:2])
            if dist < 0.1:
                print(f"  🎉 PLACED in box! dist={dist:.3f}")
            else:
                print(f"  📍 Released marker at {np.round(mpos,3)}")

    # ── SOCKET ───────────────────────────────────────────────────
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT)); sock.listen(1); sock.settimeout(0.01)
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
                # Capture virtual camera image
                scene.update_render()
                cam.take_picture()
                rgba = cam.get_picture("Color")
                img  = (rgba[:,:,:3]*255).clip(0,255).astype("uint8")

                mpos = np.array(marker.get_pose().p)
                ee_p = np.array(ee_link.get_pose().p)

                d = pickle.dumps({
                    "img":      img,
                    "n_joints": n_joints,
                    "qpos":     current_q,
                    "step":     step,
                    "marker":   mpos,
                    "ee_pos":   ee_p,
                })
                conn.sendall(len(d).to_bytes(4,'big') + d)

                # Receive EEF target
                raw=b""
                while len(raw)<4: raw+=conn.recv(4-len(raw))
                al=int.from_bytes(raw,'big'); ad=b""
                while len(ad)<al: ad+=conn.recv(al-len(ad))
                payload = pickle.loads(ad)

                target_pos = np.array(payload["eef_pos"])
                target_rot = np.array(payload["eef_rot"])
                gripper    = float(payload["gripper"])

                # Solve IK
                new_q, success = solve_ik_pinocchio(target_pos, target_rot, current_q)

                # Apply gripper separately
                g = np.clip((gripper-0.006)/(0.044-0.006), 0, 1)
                if n_joints >= 7: new_q[6] = g
                if n_joints >= 8: new_q[7] = g

                for i, jt in enumerate(joints):
                    jt.set_drive_target(float(new_q[i]))
                current_q = robot.get_qpos().copy()

                # Grasp/release logic
                try_grasp(g)
                try_release(g)

                if step % 30 == 0:
                    print(f"  step={step:4d} | IK={'✅' if success else '❌'} | "
                          f"target=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}) | "
                          f"ee=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f}) | "
                          f"grip={g:.2f} | grasped={marker_grasped}")

            except Exception as e:
                print(f"  conn err: {e}"); conn = None

        for _ in range(ACT_REPEAT): scene.step()
        scene.update_render()
        viewer.render()
        step += 1

    sock.close()


# ══════════════════════════════════════════════════════════════════
#  CLIENT — RDT2 receives sim image, predicts EEF
# ══════════════════════════════════════════════════════════════════
def run_client():
    import torch, numpy as np, socket, pickle, time, cv2
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import PeftModel

    print("="*60)
    print("  RDT2 Client — Virtual Camera Input")
    print("="*60)

    print("[1/2] Loading model...")
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
            state  = pickle.loads(d)
            img_np = state["img"]
            step   = state["step"]
            mpos   = state["marker"]
            ee_pos = state["ee_pos"]

            # Show virtual camera view
            bgr = img_np[:,:,::-1].copy()
            cv2.putText(bgr, f"Virtual Cam | inf#{nc} | step#{step}",
                        (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(bgr, f"marker:({mpos[0]:.2f},{mpos[1]:.2f},{mpos[2]:.2f})",
                        (5,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
            cv2.putText(bgr, f"ee:({ee_pos[0]:.2f},{ee_pos[1]:.2f},{ee_pos[2]:.2f})",
                        (5,58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)
            cv2.imshow("Virtual Camera — Model Input", bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            fn += 1

            if not queue:
                img_pil = Image.fromarray(img_np)
                t0 = time.time()
                acts = predict(img_pil); queue=list(acts); nc+=1
                a0 = acts[0]
                print(f"  step={step:4d} inf#{nc:3d} | "
                      f"eef=({a0[0]:.3f},{a0[1]:.3f},{a0[2]:.3f}) | "
                      f"rot=({a0[3]:.2f},{a0[4]:.2f},{a0[5]:.2f}) | "
                      f"grip={a0[6]:.3f} | t={time.time()-t0:.1f}s")

            act = queue.pop(0)
            payload={"eef_pos": act[:3], "eef_rot": act[3:6], "gripper": act[6]}
            rep=pickle.dumps(payload)
            sock.sendall(len(rep).to_bytes(4,'big')+rep)

        except Exception as e:
            print(f"  err: {e}"); break

    cv2.destroyAllWindows()
    sock.close()
    print(f"  Done — {fn} frames  {nc} inferences")


if args.mode=="server": run_server()
else: run_client()