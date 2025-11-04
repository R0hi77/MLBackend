from flask import Flask
from flask_mqtt import Mqtt
import logging
import os
from app.routes.routes import main as main_blueprint
from app.handlers.mqtt_handler import setup_mqtt_handlers
from app.routes.prediction_route import prediction
from app.utils.model_manager import NILMModelManager

mqtt = Mqtt()

def create_app():
    app = Flask(__name__)

    from app.configurations.config import Config
    app.config.from_object(Config)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    mqtt.init_app(app)

    app.logger.info(
        "MQTT config: host=%s port=%s username=%s",
        app.config.get("MQTT_BROKER_URL"),
        app.config.get("MQTT_BROKER_PORT"),
        "set" if app.config.get("MQTT_USERNAME") else "none",
    )
    setup_mqtt_handlers(mqtt)

    model_manager = NILMModelManager()
    model_path = app.config.get('MODEL_PATH')
    scaler_path = app.config.get('SCALER_PATH')

    if model_path and scaler_path:
        if model_manager.load_model(model_path,scaler_path):
            app.logger.info("NILM model loaded succesfully")
        else:
            app.logger.info("Failed to load NILM model")
    else:
        app.logger.warning("Model paths not configured")

    app.model_manager = model_manager
    
    #register blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(prediction)

    return app