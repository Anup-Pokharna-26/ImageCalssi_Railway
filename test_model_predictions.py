import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Load the model
MODEL_PATH = 'model/best_model.keras'
CLASS_NAMES_PATH = 'model/class_names.txt'

def load_class_names():
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def prepare_image_methods(filepath):
    """Test different preprocessing methods"""
    methods = {}
    
    # Method 1: Original from app.py
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    methods['original'] = np.expand_dims(img, axis=0)
    
    # Method 2: Using tf.keras preprocessing
    img2 = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img2 = tf.keras.preprocessing.image.img_to_array(img2)
    img2 = img2 / 255.0
    methods['keras_load'] = np.expand_dims(img2, axis=0)
    
    # Method 3: With normalization [-1, 1]
    img3 = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3, (224, 224))
    img3 = img3.astype("float32")
    img3 = (img3 - 127.5) / 127.5  # Normalize to [-1, 1]
    methods['normalized_-1_1'] = np.expand_dims(img3, axis=0)
    
    # Method 4: MobileNet preprocessing
    img4 = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    img4 = tf.keras.preprocessing.image.img_to_array(img4)
    img4 = tf.keras.applications.mobilenet_v2.preprocess_input(img4)
    methods['mobilenet_preprocess'] = np.expand_dims(img4, axis=0)
    
    return methods

def test_model_with_dummy_data():
    """Test model with random data to check if weights are loaded"""
    print("\n" + "="*50)
    print("Testing Model with Random Data")
    print("="*50)
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_names = load_class_names()
    
    # Test with random data
    random_input = np.random.random((1, 224, 224, 3)).astype('float32')
    
    # Get predictions
    predictions = model.predict(random_input, verbose=0)
    
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Prediction sum: {predictions.sum():.4f} (should be close to 1.0)")
    print(f"Max confidence: {predictions.max():.4f}")
    print(f"Min confidence: {predictions.min():.6f}")
    
    # Check if all predictions are similar (indicating untrained/random weights)
    std_dev = np.std(predictions[0])
    print(f"Standard deviation: {std_dev:.6f}")
    if std_dev < 0.01:
        print("⚠️ WARNING: Very low standard deviation - model might not be trained!")
    
    # Show top 5 predictions
    top_5 = np.argsort(predictions[0])[-5:][::-1]
    print("\nTop 5 predictions for random input:")
    for idx in top_5:
        print(f"  {class_names[idx]}: {predictions[0][idx]*100:.2f}%")
    
    return model, class_names

def test_with_sample_image(model, class_names):
    """Test with a sample image if available"""
    print("\n" + "="*50)
    print("Testing with Sample Images")
    print("="*50)
    
    # Look for sample images
    upload_dir = Path('static/uploads')
    sample_images = []
    
    if upload_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            sample_images.extend(upload_dir.glob(ext))
    
    if not sample_images:
        print("No sample images found in static/uploads")
        print("Creating a test image...")
        
        # Create a simple test image
        test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        # Add some patterns
        test_img[50:100, 50:100] = [255, 0, 0]  # Red square
        test_img[100:150, 100:150] = [0, 255, 0]  # Green square
        test_img[150:200, 150:200] = [0, 0, 255]  # Blue square
        
        test_path = 'test_image.jpg'
        cv2.imwrite(test_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        sample_images = [Path(test_path)]
    
    for img_path in sample_images[:3]:  # Test up to 3 images
        print(f"\nTesting: {img_path.name}")
        print("-" * 30)
        
        # Test different preprocessing methods
        methods = prepare_image_methods(str(img_path))
        
        for method_name, processed_img in methods.items():
            predictions = model.predict(processed_img, verbose=0)
            
            top_idx = np.argmax(predictions[0])
            confidence = predictions[0][top_idx] * 100
            
            print(f"\n{method_name}:")
            print(f"  Predicted: {class_names[top_idx]}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Sum of probabilities: {predictions.sum():.4f}")
            
            # Show top 3
            top_3 = np.argsort(predictions[0])[-3:][::-1]
            print("  Top 3:")
            for idx in top_3:
                print(f"    - {class_names[idx]}: {predictions[0][idx]*100:.2f}%")

def check_model_weights(model):
    """Check if model weights are properly initialized"""
    print("\n" + "="*50)
    print("Checking Model Weights")
    print("="*50)
    
    # Check a few layers
    for layer in model.layers[:10]:
        if layer.weights:
            weight = layer.weights[0]
            weight_values = weight.numpy()
            
            print(f"\nLayer: {layer.name}")
            print(f"  Shape: {weight_values.shape}")
            print(f"  Mean: {weight_values.mean():.6f}")
            print(f"  Std: {weight_values.std():.6f}")
            print(f"  Min: {weight_values.min():.6f}")
            print(f"  Max: {weight_values.max():.6f}")
            
            # Check if weights are all zeros or very small
            if np.abs(weight_values).max() < 1e-6:
                print("  ⚠️ WARNING: Weights are nearly zero!")
            elif weight_values.std() < 1e-6:
                print("  ⚠️ WARNING: Weights have no variation!")

def main():
    print("="*50)
    print("Model Diagnostic Test")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(CLASS_NAMES_PATH):
        print(f"❌ Class names not found at {CLASS_NAMES_PATH}")
        return
    
    # Get model file size
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # Convert to MB
    print(f"Model file size: {model_size:.2f} MB")
    
    if model_size < 20:
        print("⚠️ WARNING: Model file seems unusually small!")
    
    # Run tests
    model, class_names = test_model_with_dummy_data()
    check_model_weights(model)
    test_with_sample_image(model, class_names)
    
    print("\n" + "="*50)
    print("Diagnostic Complete")
    print("="*50)

if __name__ == "__main__":
    main()
