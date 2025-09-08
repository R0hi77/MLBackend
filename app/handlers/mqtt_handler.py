import json
from datetime import datetime
from flask import current_app
import logging
from app.utils.model_manager import NILMModelManager

logger = logging.getLogger(__name__)

def setup_mqtt_handlers(mqtt):

    @mqtt.on_connect()
    def handle_connect(client,userdata,flags,rc):
        if rc  == 0:
            logger.info("established connection with broker")
            mqtt.subscribe('sensor-data/#')
        else:
            logger.info(f"failed to establish connection with logger {rc}")
            

    @mqtt.on_message()
    def handle_mqtt_message(client,userdata,message):
        try:
            topic = message.topic
            topic_parts = topic.split('/')

            if topic.startswith('sensor-data/'):
                handle_sensor_data(topic_parts,message,mqtt)
            else:
                logger.warning(f"Unknow topic pattern {topic}")
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")


    @mqtt.on_disconnect()
    def handle_disconnect():
        logger.info(f"disconnected from broker")




def handle_sensor_data(topic_parts, message, mqtt):
    """Handle incoming sensor data and publish predictions"""
    try:
        # Extract device ID from topic: sensor-data/{device_id}
        if len(topic_parts) < 2:
            logger.warning("Invalid sensor-data topic format")
            return
            
        device_id = topic_parts[1]
        sensor_data = json.loads(message.payload.decode())
        
        # Extract aggregate power from sensor data
        aggregate_power = sensor_data.get('aggregate_power') or sensor_data.get('power') or sensor_data.get('aggregate')
        
        if aggregate_power is None:
            logger.warning(f"No aggregate power found in sensor data for device {device_id}")
            return
        
        # Get model manager
        model_manager = NILMModelManager()
        
        # Add data to buffer
        model_manager.add_sensor_data(device_id, float(aggregate_power))
        
        # Check if we can make a prediction
        if model_manager.can_predict(device_id):
            predictions = model_manager.predict_appliances(device_id)
            
            if predictions:
                # Publish predictions to predictions/{device_id}
                prediction_topic = f"predictions/{device_id}"
                prediction_payload = {
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions,
                    'input_power': aggregate_power
                }
                
                mqtt.publish(prediction_topic, json.dumps(prediction_payload))
                logger.info(f"Published prediction for device {device_id}")
            else:
                logger.error(f"Failed to generate predictions for device {device_id}")
        else:
            # Log buffer status for debugging
            status = model_manager.get_device_buffer_status(device_id)
            logger.debug(f"Device {device_id}: Buffer length {status['length']}/{status['required_length']}")
            
    except Exception as e:
        logger.error(f"Error handling sensor data: {e}")

