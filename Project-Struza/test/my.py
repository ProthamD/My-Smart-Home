import time
import random
import paho.mqtt.client as mqtt

# MQTT setup
broker = "localhost"
port = 1883
camera_topic = "home/sensors/camera"

client = mqtt.Client()

client.connect(broker, port)

def simulate_camera():
    while True:
        camera_status = random.choice(["Camera Active", "Camera Inactive"])
        message = f"Camera status: {camera_status}"
        client.publish(camera_topic, message)
        print(message)
        time.sleep(5)  # Publish every 5 seconds

simulate_camera()