import json
import logging
import threading
from datetime import datetime

from app.utils.model_manager import NILMModelManager
from flask import current_app

logger = logging.getLogger(__name__)

processed_messages = {}
message_lock = threading.Lock()


def setup_mqtt_handlers(mqtt):

    @mqtt.on_connect()
    def handle_connect(client, userdata, flags, rc):
        if rc == 0:
            logger.info("established connection with broker")
            mqtt.subscribe("sensor-data/#")
        else:
            logger.info(f"failed to establish connection with logger {rc}")

    @mqtt.on_message()
    def handle_mqtt_message(client, userdata, message):
        try:
            topic = message.topic
            topic_parts = topic.split("/")

            if topic.startswith("sensor-data/"):
                handle_sensor_data(topic_parts, message, mqtt)
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
        aggregate_power = (
            sensor_data.get("aggregate_power")
            or sensor_data.get("power")
            or sensor_data.get("aggregate")
        )

        if aggregate_power is None:
            logger.warning(
                f"No aggregate power found in sensor data for device {device_id}"
            )
            return

        # Create unique message ID for deduplication
        timestamp = sensor_data.get("timestamp", datetime.now().isoformat())
        message_id = f"{device_id}_{timestamp}_{aggregate_power}"

        # Check if we've already processed this message
        with message_lock:
            if message_id in processed_messages:
                logger.debug(f" Skipping duplicate message: {message_id}")
                return

            # Mark this message as processed
            processed_messages[message_id] = datetime.now()

            # Clean old entries to prevent memory growth (keep only last 100)
            if len(processed_messages) > 100:
                oldest_keys = sorted(processed_messages.keys())[:50]
                for key in oldest_keys:
                    del processed_messages[key]
                logger.debug(f"Cleaned {len(oldest_keys)} old message IDs from cache")

        # Get model manager
        model_manager = NILMModelManager()

        # Add data to buffer
        model_manager.add_sensor_data(device_id, float(aggregate_power))
        logger.debug(f"Added sensor data for device {device_id}: {aggregate_power}W")

        # Check if we can make a prediction
        if model_manager.can_predict(device_id):
            logger.info(f"Buffer ready for device {device_id}. Making prediction...")
            predictions = model_manager.predict_appliances(device_id)

            if predictions:
                # Publish predictions to predictions/{device_id}
                prediction_topic = f"predictions/{device_id}"
                prediction_payload = {
                    "device_id": device_id,
                    "timestamp": datetime.now().isoformat(),
                    "predictions": predictions,
                    "input_power": aggregate_power,
                }

                mqtt.publish(prediction_topic, json.dumps(prediction_payload))
                logger.info(f" Published prediction for device {device_id}")
            else:
                logger.error(f" Failed to generate predictions for device {device_id}")
        else:
            # Log buffer status for debugging
            status = model_manager.get_device_buffer_status(device_id)
            logger.debug(
                f"Device {device_id}: Buffer length {status['length']}/{status['required_length']} - waiting for more data"
            )

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in message payload: {e}")
    except Exception as e:
        logger.error(f"Error handling sensor data: {e}", exc_info=True)
