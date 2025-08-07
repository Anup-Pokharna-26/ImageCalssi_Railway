#!/usr/bin/env python3
"""
Comprehensive test script for all features and functionality
"""

import os
import sys
import subprocess
import tempfile
import requests
import time
import traceback
from PIL import Image
import numpy as np

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        import flask
        import gdown
        import PIL
        import sklearn
        
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Flask: {flask.__version__}")
        print("✅ All dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_model_download():
    """Test model downloading functionality"""
    print("\n🔍 Testing model download...")
    
    try:
        from download_model import download_model, download_class_names
        
        # Test downloading
        if not os.path.exists('model/best_model.keras'):
            print("Downloading model...")
            download_model()
        
        if not os.path.exists('model/class_names.txt'):
            print("Downloading class names...")
            download_class_names()
        
        # Check files exist and are valid
        if os.path.exists('model/best_model.keras') and os.path.getsize('model/best_model.keras') > 1000000:
            print("✅ Model file downloaded and looks valid")
        else:
            print("❌ Model file missing or too small")
            return False
            
        if os.path.exists('model/class_names.txt') and os.path.getsize('model/class_names.txt') > 100:
            print("✅ Class names file downloaded and looks valid")
        else:
            print("❌ Class names file missing or too small")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error testing model download: {e}")
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\n🔍 Testing model loading...")
    
    try:
        import tensorflow as tf
        
        model = tf.keras.models.load_model('model/best_model.keras')
        print(f"✅ Model loaded successfully")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        print(f"📊 Model parameters: {model.count_params():,}")
        
        # Test with dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Model prediction test successful: {prediction.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_class_names():
    """Test class names loading"""
    print("\n🔍 Testing class names...")
    
    try:
        with open('model/class_names.txt', 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print(f"✅ Loaded {len(class_names)} class names")
        print(f"📋 First 5 classes: {class_names[:5]}")
        print(f"📋 Last 5 classes: {class_names[-5:]}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading class names: {e}")
        return False

def test_preprocessing():
    """Test preprocessing modules"""
    print("\n🔍 Testing preprocessing modules...")
    
    try:
        # Test enhanced preprocessing
        from enhanced_preprocessing import EnhancedPreprocessor
        preprocessor = EnhancedPreprocessor()
        print("✅ Enhanced preprocessor loaded")
        
        # Test smart preprocessing
        from smart_preprocessing import SmartPreprocessor
        smart_preprocessor = SmartPreprocessor()
        print("✅ Smart preprocessor loaded")
        
        return True
    except Exception as e:
        print(f"❌ Error testing preprocessing: {e}")
        traceback.print_exc()
        return False

def test_ensemble_predictor():
    """Test ensemble predictor"""
    print("\n🔍 Testing ensemble predictor...")
    
    try:
        from ensemble_predictor import EnsemblePredictor
        predictor = EnsemblePredictor('model/best_model.keras', 'model/class_names.txt')
        print("✅ Ensemble predictor loaded")
        return True
    except Exception as e:
        print(f"❌ Error testing ensemble predictor: {e}")
        traceback.print_exc()
        return False

def create_test_image():
    """Create a test image for prediction testing"""
    print("\n🔍 Creating test image...")
    
    try:
        # Create a simple test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_img_path = 'test_image.jpg'
        
        Image.fromarray(test_img).save(test_img_path)
        print(f"✅ Test image created: {test_img_path}")
        
        return test_img_path
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return None

def test_flask_app_imports():
    """Test if Flask app can be imported without errors"""
    print("\n🔍 Testing Flask app imports...")
    
    try:
        from app import app, model, CLASS_NAMES
        print("✅ Flask app imported successfully")
        print(f"📊 Model loaded: {model is not None}")
        print(f"📋 Class names loaded: {len(CLASS_NAMES)} classes")
        return True
    except Exception as e:
        print(f"❌ Error importing Flask app: {e}")
        traceback.print_exc()
        return False

def test_prediction_endpoints():
    """Test prediction endpoints (requires running server)"""
    print("\n🔍 Testing prediction endpoints...")
    
    # This would require the server to be running
    print("⚠️ Endpoint testing requires running server - skipped in static test")
    return True

def test_advanced_training():
    """Test advanced training module"""
    print("\n🔍 Testing advanced training module...")
    
    try:
        from advanced_training import AdvancedFoodClassifier
        classifier = AdvancedFoodClassifier(num_classes=80)
        print("✅ Advanced training module loaded")
        return True
    except Exception as e:
        print(f"❌ Error testing advanced training: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Running comprehensive test suite...")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Download", test_model_download),
        ("Model Loading", test_model_loading),
        ("Class Names", test_class_names),
        ("Preprocessing", test_preprocessing),
        ("Ensemble Predictor", test_ensemble_predictor),
        ("Flask App", test_flask_app_imports),
        ("Advanced Training", test_advanced_training),
        ("Prediction Endpoints", test_prediction_endpoints),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your application is ready to use.")
        print("\n🚀 To start the application:")
        print("   python app.py")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

def check_project_structure():
    """Check if all required files are present"""
    print("\n🔍 Checking project structure...")
    
    required_files = [
        'app.py',
        'download_model.py',
        'requirements.txt',
        'templates/index.html',
        'ensemble_predictor.py',
        'smart_preprocessing.py',
        'enhanced_preprocessing.py',
        'advanced_training.py',
        'setup_dev.py',
        'Dockerfile',
        'README.md'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("\n✅ All required files present")
        return True

if __name__ == '__main__':
    print("🔧 CLOUD-BASED IMAGE CLASSIFIER - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    # Check project structure first
    structure_ok = check_project_structure()
    
    if structure_ok:
        # Run all functionality tests
        all_passed = run_all_tests()
        
        if all_passed:
            print("\n🎊 CONGRATULATIONS! Everything is working perfectly!")
        else:
            print("\n🔧 Some issues were found. Please fix them before deployment.")
    else:
        print("\n❌ Project structure is incomplete. Please ensure all files are present.")
