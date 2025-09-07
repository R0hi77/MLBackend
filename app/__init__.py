from flask import Flask
from flask_mqtt import Mqtt
from app.routes.routes import main as main_blueprint

mqtt = Mqtt()

def create_app():
    app = Flask(__name__)

    from app.configurations.config import Config
    app.config.from_object(Config)

    # mqtt.init_app(app)

    #register blueprints
    app.register_blueprint(main_blueprint)

    return app