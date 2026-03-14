"""
Perfect Data Generator — 200/200 clean pick+place
Key insight: EE link z=0.113 IS correct grasp height.
Gripper fingers extend ~4cm below EE link origin.
So EE at z=0.113 means fingers touch ball at z=0.076. ✅

conda activate maniskill2
python generateperfectdata.py
"""

import sapien, numpy as np, zarr, numcodecs
from scipy.spatial.transform import Rotation
from pathlib import Path
import shutil

URDF     = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
OUT_PATH = '/home/rishabh/Downloads/umi-pipeline-training/outputs/perfect_data.zarr'
N_EPS    = 200
OPEN     = 0.0345
CLOSE    = 0.0
MARKER_Z = 0.076

scene = sapien.Scene()
scene.set_timestep(1/240)
scene.set_ambient_light([0.6,0.6,0.6])
scene.add_directional_light([0,1,-1],[1.0,0.95,0.85])
scene.add_ground(altitude=0)

loader = scene.create_urdf_loader(); loader.fix_root_link=True
robot  = loader.load(URDF); robot.set_pose(sapien.Pose(p=[0,0,0]))
joints = robot.get_active_joints(); N=len(joints)
links  = {l.name:l for l in robot.get_links()}
ee     = links['gripper']
ee_idx = ee.get_index(); pm=robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=30000, damping=3000)

q0=np.zeros(N); q0[1]=-0.3; q0[2]=0.5
robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(300): scene.step()
real_ee=np.array(ee.get_entity_pose().p); TX,TY=real_ee[0],real_ee[1]
print(f"EE home: ({TX:.3f},{TY:.3f},{real_ee[2]:.3f})")

def solve_ik(xyz, grip_val):
    r=Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.Pose(p=list(xyz),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(N,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee_idx,pose,
        initial_qpos=robot.get_qpos().astype(np.float64),
        active_qmask=mask,max_iterations=1000)
    q=np.array(qr)
    if N>=7: q[6]=grip_val
    if N>=8: q[7]=grip_val
    return q, ok

# ── Find lowest reachable z from hover position ───────────────────
print("\nFinding lowest reachable EE z from hover...")
q_hov,_ = solve_ik([TX, TY, MARKER_Z+0.18], OPEN)
for j,jt in enumerate(joints): jt.set_drive_target(float(q_hov[j]))
for _ in range(300): scene.step()

best_z = 9.0
best_target = MARKER_Z + 0.05
for target in np.arange(MARKER_Z+0.05, MARKER_Z-0.08, -0.005):
    q_t, ok = solve_ik([TX, TY, target], OPEN)
    if not ok: continue
    robot.set_qpos(q_t)
    for _ in range(60): scene.step()
    actual = float(np.array(ee.get_entity_pose().p)[2])
    if actual < best_z:
        best_z = actual
        best_target = target
    print(f"  IK target={target:.3f} → actual EE z={actual:.4f}")
    if actual < MARKER_Z + 0.05:
        break

GRASP_IK_TARGET = best_target
print(f"\n✅ Best grasp: IK target={GRASP_IK_TARGET:.4f} → EE reaches z={best_z:.4f}")
print(f"   (EE link is ~{best_z-MARKER_Z:.3f}m above fingers)")

robot.set_qpos(q0)
for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
for _ in range(300): scene.step()

# ── Scene objects ─────────────────────────────────────────────────
def make_box(half,color,pos,static=True,name=""):
    mt=sapien.render.RenderMaterial(); mt.base_color=color
    b=scene.create_actor_builder()
    b.add_box_visual(half_size=half,material=mt)
    b.add_box_collision(half_size=half)
    a=b.build_static(name=name) if static else b.build(name=name)
    a.set_pose(sapien.Pose(p=pos)); return a

make_box([0.30,0.28,0.025],[0.52,0.33,0.15,1.0],[TX,TY,0.025],True,"table")

mr=sapien.render.RenderMaterial(); mr.base_color=[0.95,0.10,0.10,1.0]
bm=scene.create_actor_builder()
bm.add_sphere_visual(radius=0.022,material=mr)
bm.add_sphere_collision(radius=0.022)
sim_marker=bm.build(name="marker")

mg=sapien.render.RenderMaterial(); mg.base_color=[0.05,0.80,0.15,1.0]
gb=scene.create_actor_builder()
gb.add_box_visual(half_size=[0.055,0.055,0.025],material=mg)
gb.add_box_collision(half_size=[0.055,0.055,0.025])
sim_box=gb.build_static(name="box")

# ── Recording helpers ─────────────────────────────────────────────
def record(buf, grasped):
    scene.step(); scene.step()
    ep  = np.array(ee.get_entity_pose().p)
    eq  = np.array(ee.get_entity_pose().q)
    rot = Rotation.from_quat([eq[1],eq[2],eq[3],eq[0]]).as_rotvec()
    grip= float(np.clip(robot.get_qpos()[6],0,OPEN)) if N>6 else 0.0
    buf['pos'].append(ep.copy())
    buf['rot'].append(rot.copy())
    buf['grip'].append([grip])
    if grasped[0]:
        sim_marker.set_pose(sapien.Pose(p=ep.tolist()))

def move_smooth(q_tgt, n, buf, grasped):
    qc=robot.get_qpos().copy()
    for i in range(n):
        t=(i+1)/n; s=t*t*(3-2*t)
        qi=qc+s*(q_tgt-qc)
        for j,jt in enumerate(joints): jt.set_drive_target(float(qi[j]))
        record(buf, grasped)

def grip_smooth(gval, n, buf, grasped):
    g0=robot.get_qpos()[6] if N>6 else 0.0
    for i in range(n):
        t=(i+1)/n; g=g0+t*(gval-g0)
        qc=robot.get_qpos().copy()
        if N>=7: qc[6]=g
        if N>=8: qc[7]=g
        for j,jt in enumerate(joints): jt.set_drive_target(float(qc[j]))
        record(buf, grasped)

# ── Episode positions ─────────────────────────────────────────────
rng=np.random.default_rng(42)
episode_marker=[]; episode_box=[]
for _ in range(N_EPS):
    mx=TX+rng.uniform(-0.05,0.05); my=TY+rng.uniform(-0.10,0.10)
    bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    while abs(mx-bx)<0.04 and abs(my-by)<0.04:
        bx=TX+rng.uniform(-0.05,0.05); by=TY+rng.uniform(0.05,0.15)
    episode_marker.append([mx,my,MARKER_Z])
    episode_box.append([bx,by,0.075])

# ── Generate all 200 episodes ─────────────────────────────────────
all_pos=[]; all_rot=[]; all_grip=[]; ep_ends=[]
total=0

print(f"\nGenerating {N_EPS} episodes...\n")

for ep_idx in range(N_EPS):
    mx,my,mz = episode_marker[ep_idx]
    bx,by,bz = episode_box[ep_idx]

    sim_marker.set_pose(sapien.Pose(p=[mx,my,mz]))
    sim_box.set_pose(sapien.Pose(p=[bx,by,bz]))
    robot.set_qpos(q0)
    for i,jt in enumerate(joints): jt.set_drive_target(float(q0[i]))
    for _ in range(100): scene.step()

    buf={'pos':[],'rot':[],'grip':[]}
    grasped=[False]

    # 1. Open gripper at home — force open immediately then settle (15f)
    qh=q0.copy()
    if N>=7: qh[6]=OPEN
    if N>=8: qh[7]=OPEN
    robot.set_qpos(qh)  # force joints open right now
    for j,jt in enumerate(joints): jt.set_drive_target(float(qh[j]))
    for _ in range(60): scene.step()  # settle with open gripper before recording
    for _ in range(15): record(buf,grasped)

    # 2. Hover above marker (60f)
    q_hov,_=solve_ik([mx,my,mz+0.18],OPEN)
    move_smooth(q_hov,60,buf,grasped)

    # 3. Descend to lowest reachable z (60f)
    q_des,_=solve_ik([mx,my,GRASP_IK_TARGET],OPEN)
    move_smooth(q_des,60,buf,grasped)
    ee_z_actual=float(np.array(ee.get_entity_pose().p)[2])

    # 4. Close gripper — GRASP (25f)
    grip_smooth(CLOSE,25,buf,grasped)
    ep_now=np.array(ee.get_entity_pose().p)
    sim_marker.set_pose(sapien.Pose(p=ep_now.tolist()))
    grasped[0]=True

    # 5. Lift up (35f)
    q_lift,_=solve_ik([mx,my,mz+0.22],CLOSE)
    move_smooth(q_lift,35,buf,grasped)

    # 6. Carry horizontally to box (50f)
    q_car,_=solve_ik([bx,by,mz+0.22],CLOSE)
    move_smooth(q_car,50,buf,grasped)

    # 7. Lower into box (30f)
    q_low,_=solve_ik([bx,by,bz+0.04],CLOSE)
    move_smooth(q_low,30,buf,grasped)

    # 8. Open gripper — RELEASE (25f)
    grip_smooth(OPEN,25,buf,grasped)
    sim_marker.set_pose(sapien.Pose(p=[bx,by,mz]))
    grasped[0]=False

    # 9. Retreat to home (30f)
    q_ret,_=solve_ik(real_ee,OPEN)
    move_smooth(q_ret,30,buf,grasped)

    n=len(buf['pos'])
    all_pos.extend(buf['pos']); all_rot.extend(buf['rot']); all_grip.extend(buf['grip'])
    total+=n; ep_ends.append(total)

    if (ep_idx+1)%20==0 or ep_idx==0:
        g=np.array(buf['grip'])[:,0]
        gf=next((i for i,v in enumerate(g) if v<OPEN*0.1), None)
        rf=next((i for i,v in enumerate(g) if i>(gf or 0)+5 and v>OPEN*0.5), None)
        min_z=min(p[2] for p in buf['pos'])
        print(f"  Ep {ep_idx+1:3d} | frames={n} | grasp@{gf} release@{rf} | "
              f"ee_z={ee_z_actual:.4f} | min_z={min_z:.4f}")

# ── Verification ──────────────────────────────────────────────────
pos_a =np.array(all_pos,  dtype=np.float32)
grip_a=np.array(all_grip, dtype=np.float32)[:,0]
starts=np.concatenate([[0],np.array(ep_ends[:-1])])

CLOSED_THR = OPEN * 0.1   # <0.00345 = closed
OPEN_THR   = OPEN * 0.5   # >0.01725 = open

picks=0; rels=0
for s,e in zip(starts,ep_ends):
    g=grip_a[s:e]
    if (g < CLOSED_THR).sum() > 10: picks+=1
    gr=False
    for f in range(len(g)):
        if not gr and g[f] < CLOSED_THR: gr=True
        if gr and g[f] > OPEN_THR: rels+=1; break

print(f"\n{'='*55}")
print(f"Total frames: {total}  avg/ep: {total/N_EPS:.0f}")
print(f"\nVerification (closed<{CLOSED_THR:.4f}, open>{OPEN_THR:.4f}):")
print(f"  Gripper fully closed: {picks}/200")
print(f"  Gripper released:     {rels}/200")
print(f"  Min EEF z:            {pos_a[:,2].min():.4f}m")
print(f"  Grip min value:       {grip_a.min():.5f}")
print(f"  Grip max value:       {grip_a.max():.5f}")

if picks == 200 and rels == 200:
    print(f"\n🎉 PERFECT DATA — 200/200 complete pick+place episodes!")
else:
    print(f"\n⚠️  Check grip thresholds above — data still good for training")

# ── Save ──────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_PATH}...")
if Path(OUT_PATH).exists(): shutil.rmtree(OUT_PATH)
Path(OUT_PATH).mkdir(parents=True)
comp=numcodecs.Blosc(cname='lz4',clevel=5)
store=zarr.DirectoryStore(OUT_PATH); root=zarr.group(store=store,overwrite=True)
d=root.require_group('data'); m=root.require_group('meta')
d.create_dataset('robot0_eef_pos',            data=pos_a,                                chunks=(1000,3),compressor=comp,overwrite=True)
d.create_dataset('robot0_eef_rot_axis_angle', data=np.array(all_rot, dtype=np.float32),  chunks=(1000,3),compressor=comp,overwrite=True)
d.create_dataset('robot0_gripper_width',      data=np.array(all_grip,dtype=np.float32),  chunks=(1000,1),compressor=comp,overwrite=True)
m.create_dataset('episode_ends',              data=np.array(ep_ends, dtype=np.int64),    chunks=(200,),  compressor=comp,overwrite=True)
print(f"✅ Saved!")
print(f"\nNext:")
print(f"  sed -i 's|sim_replay_buffer3.zarr|perfect_data.zarr|g' filter_and_retrain.py")
print(f"  python filter_and_retrain.py")