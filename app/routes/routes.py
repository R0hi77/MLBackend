from flask import Blueprint, jsonify
from datetime import datetime
from app.utils import httpStatusCodes

main = Blueprint('health', __name__,url_prefix='/api/v1')

@main.get('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "ML-Backend",
        "timestamp": datetime.utcnow().isoformat()
    }), httpStatusCodes.HTTP_200_OK


@main.get('/')
def index():
    return jsonify({
        "message": "IoT ML Prediction Service",
        "endpoints": {
            "api/v1/health": "Health check endpoint",
            "api/v1/predict":"http post endpoint for making inference"
        }
    }), httpStatusCodes.HTTP_200_OK