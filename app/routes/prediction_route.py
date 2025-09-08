from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
from app.utils import httpStatusCodes
from app.utils.model_manager import NILMModelManager

prediction = Blueprint('prediction', __name__,url_prefix='/api/v1')

@prediction.post('/predict')
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "error":{
                    "code": "MISSING_PARAMETER",
                    "message":"No request parameters found",
                    "details":"requires sensor data to run inference"
                }
            }),httpStatusCodes.HTTP_400_BAD_REQUEST
        
        device_id = 12345 if data.get("device_id") is None else data.get("device_id")

        model_manager = NILMModelManager()
        if not model_manager.initialized:
            return jsonify({
                "error":{
                    "code": "INTERNAL_SERVER_ERROR",
                    "message":"model initialization error",
                    "details":"failed to initialize model on server"
                }
            }),httpStatusCodes.HTTP_500_INTERNAL_SERVER_ERROR
        
        if 'aggregate_power' in data:
            model_manager.add_sensor_data(device_id,float(data['aggregate_power']))
        
        if not model_manager.can_predict(device_id):
            status = model_manager.get_device_buffer_status(device_id)
            return jsonify({
                "error":{
                    "code":"MISSING_PARAMETER",
                    "message":"insufficient data for prediction",
                    "details":{
                        "device_id": device_id,
                        "buffer_status": status
                    }
                }
            }),httpStatusCodes.HTTP_400_BAD_REQUEST
        
        prediction = model_manager.predict_appliances(device_id)

        if prediction:
            return jsonify({
                "success": "true",
                "data": {
                    "device_id": device_id,
                    "timestamp": datetime.now().isoformat(),
                    "predictions":prediction
                }
            }),httpStatusCodes.HTTP_200_OK
        else:
            return jsonify({
                "error":{
                    "code": "INTERNAL_SERVER_ERROR",
                    "message":"prediction failed",
                    "details":{
                        "device_id":device_id
                    }

                }
            }),httpStatusCodes.HTTP_500_INTERNAL_SERVER_ERROR 
    except Exception as e:
        current_app.logger.error(f"Model info error : {e}")
        return jsonify({
                "error":{
                    "code": "INTERNAL_SERVER_ERROR",
                    "message":"error occured on server",
                    "details":{
                        "device_id":device_id,
                        "message": str(e)
                    }

                }
            }),httpStatusCodes.HTTP_500_INTERNAL_SERVER_ERROR 
