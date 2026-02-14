#!/usr/bin/env python3
"""
Diagnostic script to inspect model output structure
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import joblib

# Custom metrics (needed for loading)
def recall_m(y_true, y_pred):
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred_bin, K.floatx()))
    fn = K.sum(K.cast(y_true * (1 - y_pred_bin), K.floatx()))
    return tp / (tp + fn + K.epsilon())

def precision_m(y_true, y_pred):
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred_bin, K.floatx()))
    fp = K.sum(K.cast((1 - y_true) * y_pred_bin, K.floatx()))
    return tp / (tp + fp + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        cross_entropy_neg = -(1 - y_true) * K.log(1 - y_pred)
        weight_neg = (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        loss = weight * cross_entropy + weight_neg * cross_entropy_neg
        return K.mean(loss)
    return focal_loss_fixed

# Load model
print("Loading model...")
custom_objects = {
    'focal_loss_fixed': focal_loss(gamma=2.0, alpha=0.25),
    'precision_m': precision_m,
    'recall_m': recall_m,
    'f1_m': f1_m
}

model = tf.keras.models.load_model('models/multi_appliance_model.keras', custom_objects=custom_objects)

print("\n" + "="*80)
print("MODEL ARCHITECTURE")
print("="*80)
model.summary()

print("\n" + "="*80)
print("MODEL OUTPUTS")
print("="*80)
print(f"Number of outputs: {len(model.outputs)}")
for i, output in enumerate(model.outputs):
    print(f"Output {i}: {output.name} | Shape: {output.shape}")

# Test prediction
print("\n" + "="*80)
print("TEST PREDICTION")
print("="*80)
dummy_input = np.random.rand(1, 120, 1).astype(np.float32)
predictions = model.predict(dummy_input, verbose=0)

print(f"\nPredictions type: {type(predictions)}")
if isinstance(predictions, list):
    print(f"Number of prediction arrays: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"  Prediction {i}: shape={pred.shape}, dtype={pred.dtype}")
        print(f"    Sample values: {pred[0][:5] if len(pred[0]) > 5 else pred[0]}")
else:
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample values:\n{predictions[0]}")

print("\n" + "="*80)
print("EXPECTED vs ACTUAL")
print("="*80)
print("Expected: 14 outputs (7 appliances × 2 outputs each)")
print(f"Actual: {len(predictions) if isinstance(predictions, list) else 1} output(s)")
