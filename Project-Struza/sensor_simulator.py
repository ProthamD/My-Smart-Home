import random
import time

def get_sensor_data():
    motion_sensor = random.choice([0, 1])
    door_sensor = random.choice([0, 1])
    return {"motion": motion_sensor, "door": door_sensor}

while True:
    sensor_data = get_sensor_data()
    print("Sensor Data:", sensor_data)
    time.sleep(2)