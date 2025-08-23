import tensorflow as tf
import numpy as np

# Load the model
model_path = 'model/best_model.keras'

try:
    # Try to load the model with compile=False to avoid optimizer issues
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Number of layers: {len(model.layers)}")
    
    # Print first few layers
    print("\nFirst 5 layers:")
    for i, layer in enumerate(model.layers[:5]):
        print(f"  Layer {i}: {layer.name} - {layer.__class__.__name__}")
        if hasattr(layer, 'input_shape'):
            try:
                print(f"    Input shape: {layer.input_shape}")
            except:
                pass
    
    # Test with dummy input
    print("\nTesting with dummy input...")
    
    # Try different input shapes
    test_shapes = [
        (1, 224, 224, 3),  # Standard RGB
        (1, 224, 224, 1),  # Grayscale
        (1, 381, 381, 3),  # Different size RGB
        (1, 381, 381, 1),  # Different size grayscale
    ]
    
    for shape in test_shapes:
        try:
            dummy_input = np.random.random(shape).astype('float32')
            output = model.predict(dummy_input, verbose=0)
            print(f"  ✓ Shape {shape} works! Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Shape {shape} failed: {str(e)[:100]}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying alternative loading method...")
    
    # Try with custom objects
    try:
        with tf.keras.utils.custom_object_scope({'tf': tf}):
            model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded with custom object scope!")
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
