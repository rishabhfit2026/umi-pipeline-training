"""
UMI trajectory playback in SAPIEN
Direct linear remapping of UMI coordinates to SAPIEN workspace.
Robot will actually reach the ball and place in box.

conda activate maniskill2
python sapien_umi_playback.py
"""

import sapien, numpy as np, zarr
from scipy.spatial.transform import Rotation

ZARR_PATH = '/home/rishabh/Downloads/umi-pipeline-training/replay_buffer.zarr'
URDF      = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'

z          = zarr.open(ZARR_PATH,'r')
pos        = z['data']['robot0_eef_pos'][:]
grip       = z['data']['robot0_gripper_width'][:]
ends       = z['meta']['episode_ends'][:]
starts     = np.concatenate([[0],ends[:-1]])
umi_gmin,umi_gmax = float(grip.min()),float(grip.max())

print(f"Episodes: {len(ends)}, Frames: {len(pos)}")
print(f"UMI x=[{pos[:,0].min():.3f},{pos[:,0].max():.3f}]")
print(f"UMI y=[{pos[:,1].min():.3f},{pos[:,1].max():.3f}]")
print(f"UMI z=[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")

# ── SAPIEN ──────────────────────────────────────────────────────────
scene=sapien.Scene()
scene.set_timestep(1/240)
scene.set_ambient_light([0.5,0.5,0.5])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_directional_light([0,-1,-1],[0.3,0.3,0.4])
scene.add_ground(altitude=0)

loader=scene.create_urdf_loader(); loader.fix_root_link=True
robot=loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints=robot.get_active_joints(); N=len(joints)
links={l.name:l for l in robot.get_links()}
ee=links['gripper']; ee_idx=ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=30000,damping=3000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(300): scene.step()
real_ee=np.array(ee.get_entity_pose().p)
TX,TY,TZ=real_ee[0],real_ee[1],real_ee[2]
print(f"SAPIEN EE home: ({TX:.3f},{TY:.3f},{TZ:.3f})")

# ── Direct linear remap ─────────────────────────────────────────────
UMI_X = (pos[:,0].min(), pos[:,0].max())  # -0.766, 0.386
UMI_Y = (pos[:,1].min(), pos[:,1].max())  # -0.874, 0.291
UMI_Z = (pos[:,2].min(), pos[:,2].max())  # -0.039, 0.378

SAP_X = (0.14, 0.40)
SAP_Y = (-0.18, 0.18)
SAP_Z = (0.09, 0.72)

def remap(v, src_min, src_max, dst_min, dst_max):
    t = (v - src_min) / (src_max - src_min + 1e-9)
    t = np.clip(t, 0.0, 1.0)
    return dst_min + t*(dst_max - dst_min)

def umi_to_sapien(xyz):
    sx = remap(xyz[0], UMI_X[0], UMI_X[1], SAP_X[0], SAP_X[1])
    sy = remap(xyz[1], UMI_Y[0], UMI_Y[1], SAP_Y[0], SAP_Y[1])
    sz = remap(xyz[2], UMI_Z[0], UMI_Z[1], SAP_Z[0], SAP_Z[1])
    return np.array([sx, sy, sz])

def umi_grip_to_sapien(g):
    t=(g-umi_gmin)/(umi_gmax-umi_gmin+1e-6)
    return float(np.clip(t*0.0345, 0.0, 0.0345))

# ── IK ──────────────────────────────────────────────────────────────
def solve_ik(xyz, grip_val):
    r=Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.Pose(p=list(xyz),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=500)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q,ok

# ── Objects ─────────────────────────────────────────────────────────
def make_static(half,color,p,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name); a.set_pose(sapien.Pose(p=p)); return a

make_static([0.32,0.30,0.025],[0.55,0.36,0.18,1.0],[TX,TY,0.025],"table")
make_static([0.24,0.22,0.002],[0.95,0.95,0.90,1.0],[TX,TY,0.052],"mat")

mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.08,0.08,1.0]
bm=scene.create_actor_builder()
bm.add_sphere_visual(radius=0.022,material=mr)
bm.add_sphere_collision(radius=0.022)
sim_marker=bm.build(name="marker")

mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.82,0.18,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.020],material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.020])
sim_box=gb.build_static(name="box")

mw=sapien.render.RenderMaterial(); mw.base_color=[1.0,1.0,1.0,1.0]
ew=scene.create_actor_builder()
ew.add_sphere_visual(radius=0.012,material=mw)
ee_dot=ew.build_static(name="ee_dot")

# ── Episode playback ────────────────────────────────────────────────
def play_episode(ep_idx):
    s,e=int(starts[ep_idx]),int(ends[ep_idx])
    ep_pos =pos[s:e]
    ep_grip=grip[s:e,0]

    closed_thr = umi_gmin + (umi_gmax-umi_gmin)*0.25
    open_thr   = umi_gmin + (umi_gmax-umi_gmin)*0.65
    closed_frames = np.where(ep_grip < closed_thr)[0]
    grasp_f = closed_frames[np.argmin(ep_pos[closed_frames,2])] \
              if len(closed_frames)>0 else np.argmin(ep_pos[:,2])
    release_f = next((f for f in range(grasp_f+5,len(ep_grip))
                      if ep_grip[f]>open_thr), len(ep_pos)-1)

    ball_pos = umi_to_sapien(ep_pos[grasp_f])
    box_pos  = umi_to_sapien(ep_pos[release_f])
    ball_pos[2] = 0.076
    box_pos[2]  = 0.075

    sim_marker.set_pose(sapien.Pose(p=ball_pos.tolist()))
    sim_box.set_pose(sapien.Pose(p=box_pos.tolist()))

    print(f"\n{'='*55}")
    print(f"Ep {ep_idx+1} | frames={e-s} | grasp@{grasp_f} release@{release_f}")
    print(f"  UMI grasp:       ({ep_pos[grasp_f,0]:.3f},{ep_pos[grasp_f,1]:.3f},{ep_pos[grasp_f,2]:.3f})")
    print(f"  SAPIEN ball pos: ({ball_pos[0]:.3f},{ball_pos[1]:.3f},{ball_pos[2]:.3f})")
    print(f"  SAPIEN box pos:  ({box_pos[0]:.3f},{box_pos[1]:.3f},{box_pos[2]:.3f})")

    # Reset robot
    qh=q0.copy()
    if N>=7: qh[6]=0.0345
    if N>=8: qh[7]=0.0345
    robot.set_qpos(qh)
    for i,jt in enumerate(joints): jt.set_drive_target(float(qh[i]))
    for _ in range(100): scene.step()
    scene.update_render(); viewer.render()

    # Move to episode start
    start_sap=umi_to_sapien(ep_pos[0])
    q_s,ok=solve_ik(start_sap,0.0345)
    if ok:
        q_cur=robot.get_qpos().copy()
        for i in range(60):
            t=(i+1)/60; sm=t*t*(3-2*t)
            for j,jt in enumerate(joints):
                jt.set_drive_target(float(q_cur[j]+sm*(q_s[j]-q_cur[j])))
            for _ in range(2): scene.step()
            scene.update_render(); viewer.render()

    # Replay every frame
    prev_q=robot.get_qpos().copy()
    for f in range(len(ep_pos)):
        if viewer.closed: break
        sap_xyz =umi_to_sapien(ep_pos[f])
        sap_grip=umi_grip_to_sapien(ep_grip[f])
        q_tgt,ok=solve_ik(sap_xyz,sap_grip)
        if ok:
            q_blend=prev_q+0.6*(q_tgt-prev_q)
            for j,jt in enumerate(joints):
                jt.set_drive_target(float(q_blend[j]))
            prev_q=q_blend
        for _ in range(2): scene.step()
        ee_p=np.array(ee.get_entity_pose().p)
        ee_dot.set_pose(sapien.Pose(p=ee_p.tolist()))
        scene.update_render(); viewer.render()
        if f==grasp_f:
            print(f"  ✊ GRASP frame {f} | EE z={ee_p[2]:.3f}")
        if f==release_f:
            print(f"  🖐 RELEASE frame {f} | EE=({ee_p[0]:.3f},{ee_p[1]:.3f},{ee_p[2]:.3f})")

    print(f"  ✅ Done")
    for _ in range(150): scene.update_render(); viewer.render()

# ── Viewer ──────────────────────────────────────────────────────────
viewer=scene.create_viewer()
viewer.set_camera_xyz(TX+0.40,TY-0.35,0.50)
viewer.set_camera_rpy(0,-0.28,0.52)

n_test_start=int(len(ends)*0.8)
test_eps=list(range(n_test_start,len(ends)))

print(f"\n{'='*55}")
print(f" UMI TRAJECTORY PLAYBACK IN SAPIEN")
print(f" {len(test_eps)} real recorded episodes")
print(f" 🔴 Red ball  = grasp location")
print(f" 🟢 Green box = place location")
print(f" ⚪ White dot = live EE")
print(f"{'='*55}")

ep_ptr=0
while not viewer.closed:
    play_episode(test_eps[ep_ptr % len(test_eps)])
    ep_ptr+=1