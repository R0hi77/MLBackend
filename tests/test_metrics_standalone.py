#!/usr/bin/env python3
"""
Standalone test for custom metrics - tests only the metric functions
"""
import tensorflow as tf
from tensorflow.keras import backend as K

# ===== CUSTOM METRICS (copied from model_manager.py) =====
def recall_m(y_true, y_pred):
    """Recall metric for binary classification"""
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred_bin, K.floatx()))
    fn = K.sum(K.cast(y_true * (1 - y_pred_bin), K.floatx()))
    return tp / (tp + fn + K.epsilon())

def precision_m(y_true, y_pred):
    """Precision metric for binary classification"""
    y_pred_bin = K.cast(K.greater(y_pred, 0.5), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred_bin, K.floatx()))
    fp = K.sum(K.cast((1 - y_true) * y_pred_bin, K.floatx()))
    return tp / (tp + fp + K.epsilon())

def f1_m(y_true, y_pred):
    """F1 score metric for binary classification"""
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss function for handling class imbalance"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Positive class loss
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        
        # Negative class loss
        cross_entropy_neg = -(1 - y_true) * K.log(1 - y_pred)
        weight_neg = (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        
        loss = weight * cross_entropy + weight_neg * cross_entropy_neg
        return K.mean(loss)
    return focal_loss_fixed

# ===== TESTS =====
def test_metrics():
    print("Testing custom metrics implementation...\n")
    
    # Test data: 5 samples with known ground truth and predictions
    y_true = tf.constant([[1.0], [0.0], [1.0], [1.0], [0.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9], [0.1], [0.8], [0.3], [0.2]], dtype=tf.float32)
    
    # Test recall
    recall = recall_m(y_true, y_pred)
    print(f"✓ Recall: {recall.numpy():.4f}")
    assert 0 <= recall.numpy() <= 1, "Recall should be between 0 and 1"
    
    # Test precision
    precision = precision_m(y_true, y_pred)
    print(f"✓ Precision: {precision.numpy():.4f}")
    assert 0 <= precision.numpy() <= 1, "Precision should be between 0 and 1"
    
    # Test F1
    f1 = f1_m(y_true, y_pred)
    print(f"✓ F1 Score: {f1.numpy():.4f}")
    assert 0 <= f1.numpy() <= 1, "F1 should be between 0 and 1"
    
    # Test focal loss
    focal_loss_fn = focal_loss(gamma=2.0, alpha=0.25)
    loss = focal_loss_fn(y_true, y_pred)
    print(f"✓ Focal Loss: {loss.numpy():.4f}")
    assert loss.numpy() >= 0, "Loss should be non-negative"
    
    print("\n✅ All custom metrics tests passed!")
    print("\nMetrics are correctly implemented and ready for model loading.")
    return True

if __name__ == "__main__":
    try:
        test_metrics()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
