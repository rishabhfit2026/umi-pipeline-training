from pymycobot.myarm import MyArm
import time
mc = MyArm("/dev/ttyACM0", 115200)
time.sleep(2)
mc.power_on()
time.sleep(2)
mc.sync_send_angles([0,0,0,0,0,0,0], 20)
time.sleep(5)
print("Coords:", mc.get_coords())
print("Angles:", mc.get_angles())