import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = os.path.join(BASE_DIR,'.env')

load_dotenv(env_path)

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    MQTT_BROKER_URL = os.getenv('MQTT_BROKER_URL', 'broker.peterzzburg.online')
    MQTT_BROKER_PORT = int(os.getenv('MQTT_BROKER_PORT', 8884))
    # MQTT_USERNAME = os.getenv('MQTT_USERNAME', 'iotNodeUser')
    # MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', 'He11owor!d')

    MODEL_PATH = os.getenv('MODEL_PATH', 'models/multi_appliance_model.keras')
    SCALER_PATH = os.getenv('SCALER_PATH','models/scaler.joblib')
