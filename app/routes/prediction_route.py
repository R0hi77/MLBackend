from flask import Blueprint, jsonify, request, current_app
from datetime import datetime
from app.utils import httpStatusCodes
from app.utils.model_manager import NILMModelManager

prediction = Blueprint('prediction', __name__,url_prefix='/api/v1')

@prediction.post('/predict')
def predict():
    """HTTP endpoint for making predictions - supports single or batch data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": {
                    "code":"INVALID_PARAMETERS",
                    "message":"no request body found",
                    "details":{
                    }
                }
            }), httpStatusCodes.BAD_REQUEST
        
        device_id = data.get('device_id')
        if not device_id:
            return jsonify({
                "error":{
                    "code":"MISSING_PARAMETER",
                    "message":"expected device_id in request body",
                    "details":{}
                } 
            }), httpStatusCodes.BAD_REQUEST
        
        model_manager = NILMModelManager()
        
        # Check if model is loaded
        if not model_manager.initialized:
            return jsonify({
                "error": {
                    "code":"INTERNAL_SERVER_ERROR",
                    "message":"model not initialized",
                    "details":{

                    }
                }
            }), httpStatusCodes.INTERNAL_SERVER_ERROR
        
        # Handle batch data (array of power readings)
        if 'batch_data' in data:
            batch_data = data['batch_data']
            if not isinstance(batch_data, list):
                return jsonify({
                    "error": {
                        "code":"INVALID_FORMAT",
                        "message":"batch_data must be an array of power readings",
                        "details":{}
                    }
                }), httpStatusCodes.BAD_REQUEST
            
            # Add all batch data to buffer
            for power_reading in batch_data:
                if isinstance(power_reading, dict):
                    power_value = power_reading.get('aggregate_power') or power_reading.get('power') or power_reading.get('aggregate')
                else:
                    power_value = power_reading
                
                if power_value is not None:
                    model_manager.add_sensor_data(device_id, float(power_value))
            
            current_app.logger.info(f"Added {len(batch_data)} readings to buffer for device {device_id}")
        
        # Handle single data point
        elif 'aggregate_power' in data or 'power' in data or 'aggregate' in data:
            power_value = data.get('aggregate_power') or data.get('power') or data.get('aggregate')
            model_manager.add_sensor_data(device_id, float(power_value))
            current_app.logger.info(f"Added 1 reading to buffer for device {device_id}")
        
        else:
            return jsonify({
                "error":{
                    "code":"INVALID_FORMAT",
                    "message":"Either aggregate_power/power/aggregate or batch_data is required",
                    "details":{}
                } 
            }), httpStatusCodes.BAD_REQUEST
        
        # Check if we can make prediction
        if not model_manager.can_predict(device_id):
            status = model_manager.get_device_buffer_status(device_id)
            return jsonify({
                "error":{
                    "code":"INSUFFICIENT_DATA",
                    "message":"Insufficient data for prediction",
                    "details":{
                        'device_id': device_id,
                        'buffer_status': status
                    }
                } 
            }), httpStatusCodes.BAD_REQUEST
        
        # Make prediction
        predictions = model_manager.predict_appliances(device_id)
        
        if predictions:
            return jsonify({
                "success": True,
                "data":{
                    'device_id': device_id,
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions,
                }
            }), httpStatusCodes.HTTP_200_OK
        else:
            return jsonify({
                "error":{
                    "code":"INTERNAL_SERVER_ERROR",
                    "message":"Prediction failed",
                    "details":{
                        "device_id":device_id
                    }
                } 
            }), httpStatusCodes.INTERNAL_SERVER_ERROR
    
    except Exception as e:
        current_app.logger.error(f"Prediction error: {e}")
        return jsonify({
            "error":{
                "code":"INTERNAL_SERVER_ERROR",
                "message": "a model error occurred",
                "details":{
                    "device_id":device_id,
                    "errors":str(e)
                    }
            } 

        }), httpStatusCodes.INTERNAL_SERVER_ERROR
    

@prediction.route('/status/<device_id>', methods=['GET'])
def get_device_status(device_id):
    """Get status of a device's data buffer"""
    try:
        model_manager = NILMModelManager()
        
        if not model_manager.initialized:
            return jsonify({
                'error': 'Model not initialized'
            }), httpStatusCodes.INTERNAL_SERVER_ERROR
        
        status = model_manager.get_device_buffer_status(device_id)
        
        return jsonify({
            'device_id': device_id,
            'status': status,
            'appliances': model_manager.appliances
        }), httpStatusCodes.OK
    
    except Exception as e:
        current_app.logger.error(f"Status error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), httpStatusCodes.INTERNAL_SERVER_ERROR