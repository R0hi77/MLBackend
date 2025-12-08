# app/utils/model_manager.py
import logging
import os
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


os.environ["UCDA_VISIBLE_DEVICES"] = "-1"

# Configure TensorFlow threading BEFORE any model operations
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Configure TensorFlow threading BEFORE any model operations
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging


class NILMModelManager:
    """Singleton class to manage NILM model loading and predictions"""

    _instance = None
    _lock = threading.Lock()
    _predict_lock = threading.Lock()  # NEW: Separate lock for predictions

    # ===== CUSTOM METRICS FOR MODEL LOADING =====
    @staticmethod
    def _recall_m(y_true, y_pred):
        """Recall metric for binary classification"""
        y_pred_bin = tf.keras.backend.cast(
            tf.keras.backend.greater(y_pred, 0.5), 
            tf.keras.backend.floatx()
        )
        tp = tf.keras.backend.sum(
            tf.keras.backend.cast(y_true * y_pred_bin, tf.keras.backend.floatx())
        )
        fn = tf.keras.backend.sum(
            tf.keras.backend.cast(y_true * (1 - y_pred_bin), tf.keras.backend.floatx())
        )
        return tp / (tp + fn + tf.keras.backend.epsilon())

    @staticmethod
    def _precision_m(y_true, y_pred):
        """Precision metric for binary classification"""
        y_pred_bin = tf.keras.backend.cast(
            tf.keras.backend.greater(y_pred, 0.5), 
            tf.keras.backend.floatx()
        )
        tp = tf.keras.backend.sum(
            tf.keras.backend.cast(y_true * y_pred_bin, tf.keras.backend.floatx())
        )
        fp = tf.keras.backend.sum(
            tf.keras.backend.cast((1 - y_true) * y_pred_bin, tf.keras.backend.floatx())
        )
        return tp / (tp + fp + tf.keras.backend.epsilon())

    @staticmethod
    def _f1_m(y_true, y_pred):
        """F1 score metric for binary classification"""
        precision = NILMModelManager._precision_m(y_true, y_pred)
        recall = NILMModelManager._recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    @staticmethod
    def _focal_loss(gamma=2.0, alpha=0.25):
        """Focal loss function for handling class imbalance"""
        def focal_loss_fixed(y_true, y_pred):
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
            
            # Positive class loss
            cross_entropy = -y_true * tf.keras.backend.log(y_pred)
            weight = alpha * y_true * tf.keras.backend.pow(1 - y_pred, gamma)
            
            # Negative class loss
            cross_entropy_neg = -(1 - y_true) * tf.keras.backend.log(1 - y_pred)
            weight_neg = (1 - alpha) * (1 - y_true) * tf.keras.backend.pow(y_pred, gamma)
            
            loss = weight * cross_entropy + weight_neg * cross_entropy_neg
            return tf.keras.backend.mean(loss)
        return focal_loss_fixed

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized"):
            return

        self.model = None
        self.scaler = None
        self.appliances = [
            "Fridge-Freezer",
            "Microwave",
            "Kettle",
            "Toaster",
            "Washing_Machine",
            "Television",
            "Fan",
        ]
        self.window_size = 120  # Updated for new LSTM+Attention model
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

            # Define custom objects for model loading
            custom_objects = {
                'focal_loss_fixed': self._focal_loss(gamma=2.0, alpha=0.25),
                'precision_m': self._precision_m,
                'recall_m': self._recall_m,
                'f1_m': self._f1_m
            }

            # Load model with custom objects
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

            # Warm up the model with a dummy prediction (important!)
            # New model uses single 120-timestep input
            dummy_input = np.zeros((1, self.window_size, 1), dtype=np.float32)
            _ = self.model.predict(dummy_input, verbose=0)

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
                self.device_buffers[device_id] = deque(
                    maxlen=self.window_size
                )
                self.logger.debug(
                    f"[{device_id}] initialized new buffer with maxlen {self.window_size}"
                )

            self.device_buffers[device_id].append(aggregate_power)
            self.logger.debug(
                f"[{device_id}] Added data point: {aggregate_power}. Buffer length: {len(self.device_buffers[device_id])}"
            )

    def can_predict(self, device_id: str) -> bool:
        """Check if we have enough data to make a prediction for a device"""
        if not self.initialized:
            self.logger.debug(f"[{device_id}] Cannot predict: Model not initialized.")
            return False

        with self.buffer_lock:
            if device_id not in self.device_buffers:
                self.logger.debug(
                    f"[{device_id}] Cannot predict: No buffer found for device."
                )
                return False
            current_length = len(self.device_buffers[device_id])
            required_length = self.window_size
            can = current_length >= required_length
            self.logger.debug(
                f"[{device_id}] Buffer length: {current_length}, Required: {required_length}. Can predict: {can}"
            )
            return can

    def prepare_sequence(self, device_id: str) -> Optional[np.ndarray]:
        """Prepare input sequence for prediction (single 120-timestep window)"""
        with self.buffer_lock:
            # Get the buffer for the device
            buffer_list = self.device_buffers.get(device_id)

            # Check if we have enough data
            if (
                not self.initialized
                or buffer_list is None
                or len(buffer_list) < self.window_size
            ):
                self.logger.debug(
                    f"[{device_id}] Cannot prepare sequence. Initialized: {self.initialized}, Buffer exists: {buffer_list is not None}, Length sufficient: {len(buffer_list) >= self.window_size if buffer_list else 'N/A'}"
                )
                return None

            # Convert deque to list for processing (copy to avoid holding lock)
            buffer = list(buffer_list)

        # Scale the data (outside the lock)
        buffer_array = np.array(buffer).reshape(-1, 1)

        # Create dummy dataframe for scaling (aggregate + appliances structure)
        dummy_df = pd.DataFrame(np.zeros((len(buffer), len(self.appliances) + 1)))
        dummy_df.iloc[:, 0] = buffer  # Aggregate is first column

        scaled_data = self.scaler.transform(dummy_df)
        scaled_aggregate = scaled_data[:, 0]  # Extract scaled aggregate

        # Create single window sequence (120 timesteps)
        window = (
            scaled_aggregate[-self.window_size :]
            .reshape(1, self.window_size, 1)
            .astype(np.float32)
        )

        return window

    def predict_appliances(self, device_id: str) -> Optional[Dict]:
        """Make predictions for all appliances"""
        if not self.initialized:
            self.logger.warning(f"[{device_id}] Model not initialized for prediction.")
            return None

        self.logger.info(
            f"[{device_id}] Attempting to prepare sequence for prediction..."
        )
        sequence = self.prepare_sequence(device_id)
        if sequence is None:
            self.logger.warning(
                f"[{device_id}] Sequence preparation failed or insufficient data. Cannot predict."
            )
            return None

        try:
            self.logger.info(
                f"[{device_id}] Calling model.predict with input shape: {sequence.shape}"
            )

            # CRITICAL: Use a separate lock for model predictions to prevent race conditions
            with self._predict_lock:
                predictions = self.model.predict(sequence, verbose=0)

            # DETAILED DEBUGGING - Understanding prediction structure
            self.logger.info(
                f"[{device_id}] Predictions type: {type(predictions)}"
            )
            self.logger.info(
                f"[{device_id}] Predictions length: {len(predictions) if isinstance(predictions, (list, tuple)) else 'N/A'}"
            )
            
            # Log each prediction element
            if isinstance(predictions, (list, tuple)):
                for idx, pred in enumerate(predictions):
                    self.logger.info(
                        f"[{device_id}] predictions[{idx}]: type={type(pred)}, shape={pred.shape if hasattr(pred, 'shape') else 'N/A'}"
                    )
            else:
                self.logger.info(
                    f"[{device_id}] Single prediction shape: {predictions.shape}"
                )

            # Process predictions based on actual structure
            try:
                if isinstance(predictions, (list, tuple)) and len(predictions) == 2:
                    # Expected: predictions[0] = regression (1, 7), predictions[1] = classification (1, 7)
                    regression_outputs = predictions[0][0]  # Shape: (7,)
                    classification_outputs = predictions[1][0]  # Shape: (7,)
                elif isinstance(predictions, np.ndarray):
                    # Single array output - need to handle differently
                    self.logger.error(f"[{device_id}] Unexpected single array output: {predictions.shape}")
                    return None
                else:
                    self.logger.error(f"[{device_id}] Unknown prediction structure")
                    return None
                
                results = {}
                for i, appliance in enumerate(self.appliances):
                    # Power prediction (need to inverse transform)
                    power_pred = regression_outputs[i]
                    
                    # Create dummy array for inverse scaling
                    dummy = np.zeros((1, len(self.appliances) + 1))
                    dummy[0, i + 1] = power_pred  # +1 because Aggregate is first

                    unscaled_power = self.scaler.inverse_transform(dummy)[0, i + 1]

                    # State prediction
                    state_raw = float(classification_outputs[i])
                    state_pred = 1 if state_raw > 0.5 else 0

                    results[appliance] = {
                        "power": max(0, float(unscaled_power)),
                        "state": state_pred,
                        "confidence": state_raw,
                    }

                self.logger.info(
                    f"[{device_id}] Successfully processed predictions for {len(results)} appliances"
                )
                return results
                
            except Exception as inner_e:
                self.logger.error(
                    f"[{device_id}] Error in prediction processing: {inner_e}",
                    exc_info=True
                )
                return None

        except Exception as e:
            self.logger.error(
                f"[{device_id}] An error occurred during prediction or post-processing: {e}",
                exc_info=True,
            )
            return None

    def get_device_buffer_status(self, device_id: str) -> Dict:
        """Get status of device buffer"""
        with self.buffer_lock:
            if device_id not in self.device_buffers:
                self.logger.debug(f"[{device_id}] No buffer found for status check.")
                return {"exists": False, "length": 0, "can_predict": False}

            buffer_length = len(self.device_buffers[device_id])
            required_length = self.window_size
            can_predict_status = buffer_length >= required_length
            self.logger.debug(
                f"[{device_id}] Buffer status: length={buffer_length}, required={required_length}, can_predict={can_predict_status}"
            )
            return {
                "exists": True,
                "length": buffer_length,
                "can_predict": can_predict_status,
                "required_length": required_length,
            }
