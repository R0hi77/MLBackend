from flask import Blueprint, jsonify
from datetime import datetime
from app.utils import httpStatusCodes

prediction = Blueprint('prediction', __name__,url_prefix='/api/v1')

@prediction.post('/predict')
def predict():
    # prediction logic here
    return jsonify({
        "message": "Prediction endpoint"
    }), 