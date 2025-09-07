import json
from datetime import datetime

def setup_mqtt_handlers(mqtt):

    @mqtt.on_connect()
    def handle_connect(client,userdata,flags,rc):
        if rc  == 0:
            print("established broker connection")
            mqtt.subscribe('sensor-data/#')
        else:
            print(f"failed to connect to broker: {rc}")

    @mqtt.on_message()
    def handle_mqtt_message(client,userdata,message):
        try:
            topic = message.topic
            device_id = topic.split('/')[-1]
            sensor_data = json.loads(message.payload.decode())
        except Exception as e:
            print(f"Error processing MQTT message: {e}")

    @mqtt.on_disconnect()
    def handle_disconnect():
        print(f"disconnected from broker")
