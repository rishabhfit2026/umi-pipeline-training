from pymycobot import MyArmM
import time

# connect robot (Linux serial port)
myarmm = MyArmM("/dev/ttyACM0")

time.sleep(2)

# power on robot
myarmm.set_robot_power_on()

# Gets the current angle of all joints
angles = myarmm.get_joints_angle()
print(f"All current joint angles are: {angles}")

time.sleep(0.5)

# Get the angle of joint 1
angle = myarmm.get_joint_angle(1)
print(f"The current angle of joint 1 is {angle}")

# Get the angle of joint 2
angle = myarmm.get_joint_angle(2)
print(f"The current angle of joint 2 is {angle}")

# Get the angle of joint 3
angle = myarmm.get_joint_angle(3)
print(f"The current angle of joint 3 is {angle}")
