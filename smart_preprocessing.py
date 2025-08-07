#!/usr/bin/env python3
"""
Smart preprocessing that adapts to different image conditions
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

class SmartPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def analyze_image_quality(self, img):
        """Analyze image quality and return enhancement recommendations"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Calculate metrics
        brightness = np.mean(gray)
        contrast = np.std(gray)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        recommendations = {
            'enhance_brightness': brightness < 80 or brightness > 200,
            'enhance_contrast': contrast < 30,
            'enhance_sharpness': blur < 100,
            'brightness_level': brightness,
            'contrast_level': contrast,
            'blur_level': blur
        }
        
        return recommendations
    
    def smart_enhance(self, img):
        """Apply smart enhancement based on image analysis"""
        # Analyze image
        analysis = self.analyze_image_quality(img)
        
        # Convert to PIL for enhancement
        pil_img = Image.fromarray(img)
        
        # Brightness enhancement
        if analysis['enhance_brightness']:
            if analysis['brightness_level'] < 80:
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(1.3)
            elif analysis['brightness_level'] > 200:
                enhancer = ImageEnhance.Brightness(pil_img)
                pil_img = enhancer.enhance(0.8)
        
        # Contrast enhancement
        if analysis['enhance_contrast']:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.4)
        
        # Sharpness enhancement
        if analysis['enhance_sharpness']:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.2)
        
        # Color enhancement (always apply mildly)
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        return np.array(pil_img)
    
    def remove_background_noise(self, img):
        """Remove background noise and focus on food items"""
        # Apply bilateral filter to reduce noise while keeping edges
        filtered = cv2.bilateralFilter(img, 9, 75, 75)
        
        # Apply Gaussian blur and then sharpen
        blurred = cv2.GaussianBlur(filtered, (3, 3), 0)
        sharpened = cv2.addWeighted(filtered, 1.5, blurred, -0.5, 0)
        
        return sharpened
    
    def crop_to_center_object(self, img):
        """Intelligently crop to focus on the main food object"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour (assumed to be the main food item)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = img[y:y+h, x:x+w]
            
            # Only return cropped if it's reasonable size
            if cropped.shape[0] > 50 and cropped.shape[1] > 50:
                return cropped
        
        # Return original if cropping didn't work well
        return img
    
    def normalize_for_model(self, img, model_type='efficientnet'):
        """Normalize image based on model requirements"""
        if model_type == 'efficientnet':
            # EfficientNet preprocessing
            img = img.astype('float32')
            img = tf.keras.applications.efficientnet.preprocess_input(img)
        elif model_type == 'resnet':
            # ResNet preprocessing
            img = img.astype('float32')
            img = tf.keras.applications.resnet50.preprocess_input(img)
        else:
            # Standard normalization
            img = img.astype('float32') / 255.0
        
        return img
    
    def preprocess_advanced(self, image_path, model_type='efficientnet'):
        """Advanced preprocessing pipeline"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Step 1: Remove background noise
        img = self.remove_background_noise(img)
        
        # Step 2: Smart enhancement
        img = self.smart_enhance(img)
        
        # Step 3: Intelligent cropping
        img = self.crop_to_center_object(img)
        
        # Step 4: Resize with aspect ratio preservation
        img = self.resize_with_aspect_ratio(img)
        
        # Step 5: Model-specific normalization
        img = self.normalize_for_model(img, model_type)
        
        # Step 6: Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def resize_with_aspect_ratio(self, img):
        """Resize while maintaining aspect ratio"""
        h, w = img.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image with mean color
        mean_color = np.mean(img, axis=(0, 1)).astype(int)
        padded_img = np.full((target_h, target_w, 3), mean_color, dtype=img.dtype)
        
        # Calculate padding
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # Place resized image in center
        padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img
        
        return padded_img
    
    def create_multiple_versions(self, image_path):
        """Create multiple versions of the same image for ensemble prediction"""
        # Load original image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        versions = []
        
        # Version 1: Standard preprocessing
        v1 = cv2.resize(img, self.target_size)
        v1 = v1.astype('float32') / 255.0
        versions.append(v1)
        
        # Version 2: Enhanced version
        v2 = self.smart_enhance(img)
        v2 = cv2.resize(v2, self.target_size)
        v2 = v2.astype('float32') / 255.0
        versions.append(v2)
        
        # Version 3: Cropped and enhanced
        v3 = self.crop_to_center_object(img)
        v3 = self.smart_enhance(v3)
        v3 = cv2.resize(v3, self.target_size)
        v3 = v3.astype('float32') / 255.0
        versions.append(v3)
        
        # Version 4: Noise reduced
        v4 = self.remove_background_noise(img)
        v4 = cv2.resize(v4, self.target_size)
        v4 = v4.astype('float32') / 255.0
        versions.append(v4)
        
        return np.array(versions)

def test_smart_preprocessing():
    """Test the smart preprocessing"""
    preprocessor = SmartPreprocessor()
    
    test_image = 'static/uploads/biryani2.jpg'  # The problematic image
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        return
    
    try:
        # Test advanced preprocessing
        processed = preprocessor.preprocess_advanced(test_image)
        print(f"✅ Advanced preprocessing successful: {processed.shape}")
        
        # Test multiple versions
        versions = preprocessor.create_multiple_versions(test_image)
        print(f"✅ Multiple versions created: {versions.shape}")
        
    except Exception as e:
        print(f"❌ Error in smart preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import os
    test_smart_preprocessing()
