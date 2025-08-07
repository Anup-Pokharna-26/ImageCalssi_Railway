#!/usr/bin/env python3
"""
Enhanced preprocessing module for ensemble prediction
"""

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance

class EnhancedPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def preprocess_single_image(self, image_path, method='standard'):
        """Preprocess a single image using specified method"""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if method == 'standard':
            return self._standard_preprocessing(img)
        elif method == 'enhanced':
            return self._enhanced_preprocessing(img)
        elif method == 'contrast_enhanced':
            return self._contrast_enhanced_preprocessing(img)
        else:
            return self._standard_preprocessing(img)
    
    def _standard_preprocessing(self, img):
        """Standard preprocessing"""
        img = cv2.resize(img, self.target_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def _enhanced_preprocessing(self, img):
        """Enhanced preprocessing with brightness and contrast adjustments"""
        # Convert to PIL for enhancement
        pil_img = Image.fromarray(img)
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        # Enhance color
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Convert back to numpy
        img = np.array(pil_img)
        
        # Resize and normalize
        img = cv2.resize(img, self.target_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
    
    def _contrast_enhanced_preprocessing(self, img):
        """Contrast enhanced preprocessing"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Resize and normalize
        img = cv2.resize(img, self.target_size)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=0)
