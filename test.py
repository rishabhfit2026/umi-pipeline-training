from pymycobot.myarm import MyArm
import time

mc = MyArm("/dev/ttyACM0",115200)
mc.power_on()
time.sleep(2)
print(mc.get_coords())

