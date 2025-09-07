from flask import Flask
from flask_mqtt import Mqtt
from app.routes.routes import main as main_blueprint
from app.handlers.mqtt_handler import setup_mqtt_handlers
from app.routes.prediction_route import prediction

mqtt = Mqtt()

def create_app():
    app = Flask(__name__)

    from app.configurations.config import Config
    app.config.from_object(Config)

    mqtt.init_app(app)
    setup_mqtt_handlers(mqtt)

    #register blueprints
    app.register_blueprint(main_blueprint)
    app.register_blueprint(prediction)

    return app