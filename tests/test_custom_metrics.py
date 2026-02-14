#!/usr/bin/env python3
"""
Quick test to verify custom metrics work correctly
"""
import sys
import os
import numpy as np
import tensorflow as tf

# Add MLBackend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MLBackend.app.utils.model_manager import NILMModelManager


def test_custom_metrics():
    """Test that custom metrics return valid outputs"""
    print("Testing custom metrics...")
    
    # Create dummy tensors
    y_true = tf.constant([[1.0], [0.0], [1.0], [1.0], [0.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.9], [0.1], [0.8], [0.3], [0.2]], dtype=tf.float32)
    
    # Test recall
    recall = NILMModelManager._recall_m(y_true, y_pred)
    print(f"✓ Recall: {recall.numpy():.4f}")
    assert 0 <= recall.numpy() <= 1, "Recall should be between 0 and 1"
    
    # Test precision
    precision = NILMModelManager._precision_m(y_true, y_pred)
    print(f"✓ Precision: {precision.numpy():.4f}")
    assert 0 <= precision.numpy() <= 1, "Precision should be between 0 and 1"
    
    # Test F1
    f1 = NILMModelManager._f1_m(y_true, y_pred)
    print(f"✓ F1 Score: {f1.numpy():.4f}")
    assert 0 <= f1.numpy() <= 1, "F1 should be between 0 and 1"
    
    # Test focal loss
    focal_loss_fn = NILMModelManager._focal_loss(gamma=2.0, alpha=0.25)
    loss = focal_loss_fn(y_true, y_pred)
    print(f"✓ Focal Loss: {loss.numpy():.4f}")
    assert loss.numpy() >= 0, "Loss should be non-negative"
    
    print("\n✅ All custom metrics tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_custom_metrics()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
