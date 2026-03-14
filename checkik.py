import sapien.core as sapien
import sapien
import numpy as np
from scipy.spatial.transform import Rotation

URDF = '/home/rishabh/Downloads/myarm_m750_fixed.urdf'
scene = sapien.core.Scene()
scene.set_timestep(1/120)
scene.add_ground(altitude=0)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(URDF)
robot.set_pose(sapien.core.Pose(p=[0,0,0]))
joints = robot.get_active_joints()
n = len(joints)
links = {l.name: l for l in robot.get_links()}
ee = links['gripper']
pm = robot.create_pinocchio_model()
for jt in joints: jt.set_drive_property(stiffness=15000, damping=1500)

q_home = np.zeros(n); q_home[1]=-0.3; q_home[2]=0.5
robot.set_qpos(q_home)
for _ in range(200): scene.step()
ep = ee.get_entity_pose()
print(f"EE home: x={ep.p[0]:.4f} y={ep.p[1]:.4f} z={ep.p[2]:.4f}")

def ik(pos, q_cur):
    r=Rotation.from_euler('xyz',[np.pi,0,0]); qv=r.as_quat()
    pose=sapien.core.Pose(p=list(pos),q=[qv[3],qv[0],qv[1],qv[2]])
    mask=np.ones(n,dtype=np.int32); mask[6:]=0
    qr,ok,_=pm.compute_inverse_kinematics(ee.get_index(),pose,
        initial_qpos=q_cur.astype(np.float64),active_qmask=mask,
        max_iterations=400,dt=0.1,damp=1e-6)
    return np.array(qr),ok

print("Testing IK at z=0.076:")
for x,y in [(0.25,0.00),(0.20,0.00),(0.20,0.08),(0.20,-0.08),(0.25,0.08),(0.25,-0.08),(0.30,0.00)]:
    tq,ok = ik([x,y,0.076], robot.get_qpos().copy())
    robot.set_qpos(tq)
    for _ in range(80): scene.step()
    ep2 = ee.get_entity_pose()
    print(f"  ({x:.2f},{y:.2f}) ok={ok} ee=({ep2.p[0]:.3f},{ep2.p[1]:.3f},{ep2.p[2]:.3f})")
    robot.set_qpos(q_home)
    for _ in range(40): scene.step()