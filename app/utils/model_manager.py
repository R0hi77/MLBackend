# app/utils/model_manager.py
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os
from collections import deque
import threading
from typing import Dict, List, Optional, Tuple
import logging

class NILMModelManager:
    """Singleton class to manage NILM model loading and predictions"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        
        self.model = None
        self.scaler = None
        self.appliances = ['Fridge-Freezer', 'Microwave', 'Kettle', 'Toaster', 'Washing_Machine', 'Television']
        self.window_size = 30
        self.t1_window = 15
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
        # Circular buffers to store recent data for each device
        self.device_buffers = {}
        self.buffer_lock = threading.Lock()
    
    def load_model(self, model_path: str, scaler_path: str) -> bool:
        """Load the TensorFlow model and scaler"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            if not os.path.exists(scaler_path):
                self.logger.error(f"Scaler file not found: {scaler_path}")
                return False
            
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.initialized = True
            
            self.logger.info("Model and scaler loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def add_sensor_data(self, device_id: str, aggregate_power: float) -> None:
        """Add new sensor data to the device buffer"""
        with self.buffer_lock:
            if device_id not in self.device_buffers:
                # Initialize buffer with enough capacity for predictions
                self.device_buffers[device_id] = deque(maxlen=self.window_size + self.t1_window)
                self.logger.debug(f"[{device_id}] initialized new buffer with maxle  {self.window_size + self.t1_window}")
            
            self.device_buffers[device_id].append(aggregate_power)
            self.logger.debug(f"[{device_id}] Added data point: {aggregate_power}. Buffer length: {len(self.device_buffers[device_id])}")
    
    def can_predict(self, device_id: str) -> bool:
        """Check if we have enough data to make a prediction for a device"""
        if not self.initialized:
            self.logger.debug(f"[{device_id}] Cannot predict: Model not initialized.")
            return False
        
        with self.buffer_lock:
            if device_id not in self.device_buffers:
                self.logger.debug(f"[{device_id}] Cannot predict: No buffer found for device.")
                return False
            current_length = len(self.device_buffers[device_id])
            required_length = self.window_size + self.t1_window
            can = current_length >= required_length
            self.logger.debug(f"[{device_id}] Buffer length: {current_length}, Required: {required_length}. Can predict: {can}")
            return can
    

    def prepare_sequence(self, device_id: str) -> Optional[Dict]:
        """Prepare input sequences for prediction"""
        with self.buffer_lock:
            # --- START: MODIFIED CODE ---
            
            # Get the buffer for the device
            buffer_list = self.device_buffers.get(device_id)
            
            # Check if we have enough data directly, instead of calling can_predict
            if not self.initialized or buffer_list is None or len(buffer_list) < (self.window_size + self.t1_window):
                self.logger.debug(f"[{device_id}] Cannot prepare sequence. Initialized: {self.initialized}, Buffer exists: {buffer_list is not None}, Length sufficient: {len(buffer_list) >= (self.window_size + self.t1_window) if buffer_list else 'N/A'}")
                return None
            
            # Convert deque to list for processing
            buffer = list(buffer_list)
            
            # --- END: MODIFIED CODE ---

            # Scale the data
            buffer_array = np.array(buffer).reshape(-1, 1)
            
            # Create dummy dataframe for scaling (aggregate + appliances structure)
            dummy_df = pd.DataFrame(np.zeros((len(buffer), len(self.appliances) + 1)))
            dummy_df.iloc[:, 0] = buffer  # Aggregate is first column
            
            scaled_data = self.scaler.transform(dummy_df)
            scaled_aggregate = scaled_data[:, 0]  # Extract scaled aggregate
            
            # Create sequences
            main_window = scaled_aggregate[-self.window_size:].reshape(1, self.window_size, 1)
            t1_window = scaled_aggregate[-(self.window_size + self.t1_window):-self.window_size].reshape(1, self.t1_window, 1)
            
            return {
                'main_window_input': main_window,
                't1_window_input': t1_window
            }
        
    
    def predict_appliances(self, device_id: str) -> Optional[Dict]:
        """Make predictions for all appliances"""
        if not self.initialized:
            self.logger.warning(f"[{device_id}] Model not initialized for prediction.")
            return None
        
        self.logger.info(f"[{device_id}] Attempting to prepare sequence for prediction...")
        sequence = self.prepare_sequence(device_id)
        if sequence is None:
            self.logger.warning(f"[{device_id}] Sequence preparation failed or insufficient data. Cannot predict.")
            return None
        
        try:
            self.logger.info(f"[{device_id}] Calling model.predict with input shapes: main={sequence['main_window_input'].shape}, t1={sequence['t1_window_input'].shape}")
            
            # This is the line where we suspect the hang is occurring
            predictions = self.model.predict(sequence, verbose=0)
            
            self.logger.info(f"[{device_id}] Model.predict returned successfully. Number of outputs: {len(predictions)}")
            
            # Process predictions
            results = {}
            for i, appliance in enumerate(self.appliances):
                # Power prediction (need to inverse transform)
                power_pred = predictions[2*i][0][0]
                self.logger.debug(f"[{device_id}] Raw power pred for {appliance}: {power_pred}")

                # Create dummy array for inverse scaling
                dummy = np.zeros((1, len(self.appliances) + 1))
                dummy[0, i + 1] = power_pred  # +1 because Aggregate is first
                
                try:
                    unscaled_power = self.scaler.inverse_transform(dummy)[0, i + 1]
                    self.logger.debug(f"[{device_id}] Unscaled power for {appliance}: {unscaled_power}")
                except Exception as e:
                    self.logger.error(f"[{device_id}] Error during scaler.inverse_transform for {appliance}: {e}")
                    unscaled_power = 0.0 # Fallback

                # State prediction
                state_raw = predictions[2*i + 1][0][0]
                state_pred = 1 if state_raw > 0.5 else 0
                self.logger.debug(f"[{device_id}] Raw state pred for {appliance}: {state_raw}, Final state: {state_pred}")

                results[appliance] = {
                    'power': max(0, unscaled_power),  # Ensure non-negative
                    'state': state_pred,
                    'confidence': float(state_raw)
                }
            
            self.logger.info(f"[{device_id}] Prediction processing complete. Returning results.")
            return results
            
        except Exception as e:
            self.logger.error(f"[{device_id}] An error occurred during prediction or post-processing: {e}", exc_info=True) # exc_info to get traceback
            return None
            
    
def get_device_buffer_status(self, device_id: str) -> Dict:
        """Get status of device buffer"""
        with self.buffer_lock:
            if device_id not in self.device_buffers:
                self.logger.debug(f"[{device_id}] No buffer found for status check.")
                return {'exists': False, 'length': 0, 'can_predict': False}
            
            buffer_length = len(self.device_buffers[device_id])
            required_length = self.window_size + self.t1_window
            can_predict_status = buffer_length >= required_length
            self.logger.debug(f"[{device_id}] Buffer status: length={buffer_length}, required={required_length}, can_predict={can_predict_status}")
            return {
                'exists': True,
                'length': buffer_length,
                'can_predict': can_predict_status,
                'required_length': required_length
            }