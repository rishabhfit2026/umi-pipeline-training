"""
Step8 Server — clean rewrite
conda activate maniskill2
python step8_server.py
"""
import sapien
import numpy as np
import socket, pickle
from scipy.spatial.transform import Rotation

URDF      = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
CHECKPOINT= '/home/rishabh/Downloads/umi-pipeline-training/outputs/rdt2-m750-sim2/checkpoint-final'
HOST, PORT= '127.0.0.1', 9997
IMG_SIZE  = 224
ACT_REPEAT= 8

# ── Scene ────────────────────────────────────────────────────────
scene = sapien.Scene()
scene.set_timestep(1/120)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1,1,1])
scene.add_ground(altitude=0)

# ── Robot ────────────────────────────────────────────────────────
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot  = loader.load(URDF)
robot.set_pose(sapien.Pose(p=[0,0,0]))
joints   = robot.get_active_joints()
N        = len(joints)
links    = {l.name:l for l in robot.get_links()}
ee       = links['gripper']
ee_idx   = ee.get_index()
pm       = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=8000, damping=800)

q0 = np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))

# Settle robot
for _ in range(300): scene.step()
real_ee = np.array(ee.get_pose().p)
TX, TY  = real_ee[0], real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f},{real_ee[2]:.3f})")

# ── Environment ──────────────────────────────────────────────────
def make_box(half, color, pos, static=True, name=""):
    mt = sapien.render.RenderMaterial()
    mt.base_color = color
    b = scene.create_actor_builder()
    b.add_box_visual(half_size=half, material=mt)
    b.add_box_collision(half_size=half)
    a = b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos))
    return a

# Table + mat
make_box([0.30,0.28,0.025], [0.52,0.33,0.15,1.0], [TX,TY,0.025],  True,  "table")
make_box([0.20,0.18,0.002], [0.96,0.96,0.94,1.0], [TX,TY,0.052],  True,  "mat")
# Table legs
for lx,ly in [(TX+0.27,TY+0.23),(TX+0.27,TY-0.23),(TX-0.27,TY+0.23),(TX-0.27,TY-0.23)]:
    make_box([0.02,0.02,0.012],[0.38,0.22,0.10,1.0],[lx,ly,0.012], True,  "leg")

# Red marker — on mat surface (z=0.054+capsule_r)
mr = sapien.render.RenderMaterial(); mr.base_color=[0.95,0.1,0.1,1.0]
bm = scene.create_actor_builder()
bm.add_capsule_visual(radius=0.015, half_length=0.040, material=mr)
sim_marker = bm.build(name="marker")
MARKER_Z = 0.069   # mat top (0.054) + radius (0.015)
sim_marker.set_pose(sapien.Pose(p=[TX, TY, MARKER_Z]))

# Green box — on mat surface
gb = scene.create_actor_builder()
mg = sapien.render.RenderMaterial(); mg.base_color=[0.05,0.75,0.15,1.0]
gb.add_box_visual(half_size=[0.05,0.05,0.022], material=mg)
gb.add_box_collision(half_size=[0.05,0.05,0.022])
sim_box = gb.build_static(name="box")
sim_box.set_pose(sapien.Pose(p=[TX, TY+0.12, 0.076]))

# ── Virtual camera ───────────────────────────────────────────────
# EXACTLY matching step10 training: top-down at (TX,TY,0.85), fovy=60deg
# SAPIEN quaternion for looking straight down (−Z world = camera +Z):
# rotation = 90deg around X axis = [cos45, sin45, 0, 0] = [0.707, 0.707, 0, 0]
cam_ent = sapien.Entity()
cam     = sapien.render.RenderCameraComponent(IMG_SIZE, IMG_SIZE)
cam.set_fovy(float(np.deg2rad(60)), True)
cam.near = 0.05
cam.far  = 10.0
cam_ent.add_component(cam)
cam_ent.set_pose(sapien.Pose(
    p=[TX, TY, 0.85],
    q=[ 0, 0.7071068, 0.7071068, 0]   # look straight down — step10 exact
))
scene.add_entity(cam_ent)

# ── IK ───────────────────────────────────────────────────────────
def solve_ik(xyz, grip_val, q_cur):
    r    = Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose = sapien.Pose(p=list(xyz), q=[qv[3],qv[0],qv[1],qv[2]])
    mask = np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_ = pm.compute_inverse_kinematics(ee_idx, pose,
        initial_qpos=q_cur.astype(np.float64),
        active_qmask=mask, max_iterations=300)
    q = np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q, ok

# ── Viewer ───────────────────────────────────────────────────────
viewer = scene.create_viewer()
viewer.set_camera_xyz(0.8,-0.8,1.2)
viewer.set_camera_rpy(0,-0.55,0.65)

# Render one frame so scene is populated
scene.update_render()
viewer.render()

# ── Socket ───────────────────────────────────────────────────────
import socket as sock_mod
srv = sock_mod.socket()
srv.setsockopt(sock_mod.SOL_SOCKET, sock_mod.SO_REUSEADDR, 1)
srv.bind((HOST,PORT)); srv.listen(1); srv.settimeout(0.01)
print(f"\nListening on {HOST}:{PORT}")
print(">>> Start client in Terminal 2: python step8_client.py <<<\n")

conn=None; step=0; q_cur=q0.copy(); grasped=False

while not viewer.closed:
    if conn is None:
        try:
            conn,addr = srv.accept()
            conn.settimeout(5.0)
            print(f"Client connected: {addr}")
        except: pass

    if conn is not None:
        try:
            # Render virtual camera
            scene.update_render()
            cam.take_picture()
            rgba = cam.get_picture('Color')
            img  = (np.clip(rgba[:,:,:3],0,1)*255).astype(np.uint8)

            ee_p     = np.array(ee.get_pose().p)
            mpos_now = np.array(sim_marker.get_pose().p)

            # Send state
            data = pickle.dumps({
                "img":    img,
                "ee_pos": ee_p,
                "marker_pos": mpos_now,
                "grasped": grasped,
                "step":   step,
            })
            conn.sendall(len(data).to_bytes(4,'big') + data)

            # Receive action
            raw=b""
            while len(raw)<4: raw+=conn.recv(4-len(raw))
            al=int.from_bytes(raw,'big'); ad=b""
            while len(ad)<al: ad+=conn.recv(al-len(ad))
            act = pickle.loads(ad)

            tpos  = np.array(act["eef_pos"])
            grip  = float(act["gripper"])          # 0=closed, 0.0345=open
            g_val = np.clip(grip/0.0345, 0, 1) * 0.0345

            new_q, ok = solve_ik(tpos, g_val, q_cur)
            for i,jt in enumerate(joints): jt.set_drive_target(float(new_q[i]))
            q_cur = robot.get_qpos().copy()

            # Grasp / release logic
            dist = np.linalg.norm(ee_p - mpos_now)
            if not grasped and grip < 0.005 and dist < 0.12:
                grasped = True
                print(f"  ✅ GRASPED! dist={dist:.3f}")
            if grasped:
                sim_marker.set_pose(sapien.Pose(p=ee_p.tolist()))
            if grasped and grip > 0.02:
                grasped = False
                bp = np.array(sim_box.get_pose().p)
                mp = np.array(sim_marker.get_pose().p)
                if np.linalg.norm(mp[:2]-bp[:2]) < 0.12:
                    print("  🎉 PLACED in box!")
                else:
                    print(f"  📍 Released at {np.round(mp,3)}")

            if step%30==0:
                print(f"  step={step:4d} IK={'✅' if ok else '❌'} "
                      f"ee=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f}) "
                      f"grip={grip:.4f} grasped={grasped}")

        except Exception as e:
            print(f"  conn err: {e}"); conn=None

    for _ in range(ACT_REPEAT): scene.step()
    scene.update_render(); viewer.render()
    step+=1

srv.close()