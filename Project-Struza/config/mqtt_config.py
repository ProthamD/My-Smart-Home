# In config/mqtt_config.py
import paho.mqtt.client as mqtt

def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("Successfully connected to broker")
    else:
        print(f"Connection failed with code {reason_code}")

# This is the critical change ▼
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) 
client.on_connect = on_connect