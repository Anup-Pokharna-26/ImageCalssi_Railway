import requests
import os
import json
from pathlib import Path
import time

# API endpoints
BASE_URL = "http://localhost:8080"
PREDICT_URL = f"{BASE_URL}/predict"
PREDICT_ENHANCED_URL = f"{BASE_URL}/predict_enhanced"
PREDICT_SMART_URL = f"{BASE_URL}/predict_smart"

def test_health():
    """Test if the server is running"""
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
    except:
        print("❌ Server is not running. Please start the Flask app first.")
        return False
    return False

def find_test_images():
    """Find test images in the uploads folder"""
    upload_dir = Path('static/uploads')
    test_images = []
    
    if upload_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(upload_dir.glob(ext))
    
    if not test_images:
        print("No test images found. Using a default test image.")
        # Use the test image we created earlier
        if os.path.exists('test_image.jpg'):
            test_images = [Path('test_image.jpg')]
    
    return test_images

def test_prediction(image_path, endpoint_url, endpoint_name):
    """Test a single prediction endpoint"""
    print(f"\n{'='*50}")
    print(f"Testing {endpoint_name}")
    print(f"Image: {image_path.name}")
    print('='*50)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            response = requests.post(endpoint_url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Handle error responses
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                return False
            
            # Display main prediction
            print(f"✅ Prediction successful!")
            print(f"  Class: {result.get('class', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}%")
            
            # Display top predictions if available
            if 'top_predictions' in result:
                print("\n  Top Predictions:")
                for pred in result['top_predictions'][:3]:
                    print(f"    - {pred['class']}: {pred['confidence']:.2f}%")
            
            # Display debug info if available
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"\n  Debug Info:")
                print(f"    Total classes: {debug.get('total_classes', 'N/A')}")
                print(f"    Prediction sum: {debug.get('prediction_sum', 'N/A')}")
                print(f"    Model confident: {debug.get('model_confident', 'N/A')}")
            
            # Display enhanced results if available
            if 'enhanced_results' in result:
                enhanced = result['enhanced_results']
                if 'uncertainty_analysis' in enhanced:
                    uncertainty = enhanced['uncertainty_analysis']
                    print(f"\n  Uncertainty Analysis:")
                    print(f"    Mean confidence: {uncertainty.get('mean_confidence', 0):.2f}%")
                    print(f"    Std deviation: {uncertainty.get('std_confidence', 0):.2f}%")
                    
                if 'recommendation' in enhanced:
                    rec = enhanced['recommendation']
                    print(f"\n  Recommendation:")
                    print(f"    {rec.get('action', 'N/A')}")
                    print(f"    Confidence level: {rec.get('confidence_level', 'N/A')}")
            
            # Display smart analysis if available
            if 'smart_analysis' in result:
                smart = result['smart_analysis']
                print(f"\n  Smart Analysis:")
                print(f"    Consensus class: {smart.get('consensus_class', 'N/A')}")
                print(f"    Consensus ratio: {smart.get('consensus_ratio', 0):.2f}")
                print(f"    Reliability: {smart.get('reliability', 'N/A')}")
                print(f"    Recommendation: {smart.get('recommendation', 'N/A')}")
            
            return True
            
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("="*60)
    print("Flask API Test Suite")
    print("="*60)
    
    # Wait a moment for server to fully start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Check if server is running
    if not test_health():
        print("\nPlease start the Flask app first with: python app.py")
        return
    
    # Find test images
    test_images = find_test_images()
    if not test_images:
        print("No test images available")
        return
    
    print(f"\nFound {len(test_images)} test image(s)")
    
    # Test each endpoint with the first image
    test_image = test_images[0]
    
    # Test standard prediction
    success = test_prediction(test_image, PREDICT_URL, "Standard Prediction")
    
    # Test enhanced prediction
    # Note: This might fail if ensemble predictor is not initialized
    success = test_prediction(test_image, PREDICT_ENHANCED_URL, "Enhanced Prediction")
    
    # Test smart prediction
    success = test_prediction(test_image, PREDICT_SMART_URL, "Smart Prediction")
    
    # Test multiple images with standard prediction
    if len(test_images) > 1:
        print("\n" + "="*60)
        print("Testing Multiple Images")
        print("="*60)
        
        for img in test_images[:5]:  # Test up to 5 images
            print(f"\nTesting: {img.name}")
            with open(img, 'rb') as f:
                files = {'file': (img.name, f, 'image/jpeg')}
                response = requests.post(PREDICT_URL, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'error' not in result:
                    print(f"  ✅ {result.get('class', 'N/A')}: {result.get('confidence', 0):.2f}%")
                else:
                    print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ❌ Failed with status: {response.status_code}")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()
