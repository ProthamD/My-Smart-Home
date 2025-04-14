import random
import time
import json
import paho.mqtt.client as mqtt

def get_sensor_data():
    motion_sensor = random.choice([0, 1])
    door_sensor = random.choice([0, 1])
    return {"motion": motion_sensor, "door": door_sensor}

client = mqtt.Client()
client.connect("localhost", 1883)  # Use Mosquitto running locally

while True:
    data = get_sensor_data()
    payload = json.dumps(data)
    client.publish("home/security", payload)
    print("Published:", payload)
    time.sleep(2)