from pymycobot.myarm import MyArm
import time

mc = MyArm("/dev/ttyACM0",115200)


time.sleep(2)
mc.power_on()

print("Coords:", mc.get_coords())
