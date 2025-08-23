#!/usr/bin/env python3
"""
Fixed ensemble prediction system that accepts an already loaded model
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from enhanced_preprocessing import EnhancedPreprocessor
from collections import Counter

class EnsemblePredictorFixed:
    def __init__(self, model, class_names):
        """Initialize with already loaded model and class names"""
        self.model = model
        self.class_names = class_names
        self.preprocessor = EnhancedPreprocessor()
        
    def predict_single_method(self, image_path, method='standard'):
        """Predict using a single preprocessing method"""
        processed_img = self.preprocessor.preprocess_single_image(image_path, method=method)
        prediction = self.model.predict(processed_img, verbose=0)
        return prediction[0]
    
    def predict_ensemble(self, image_path, methods=['standard', 'enhanced', 'contrast_enhanced']):
        """Predict using ensemble of preprocessing methods"""
        predictions = []
        
        for method in methods:
            try:
                pred = self.predict_single_method(image_path, method=method)
                predictions.append(pred)
            except Exception as e:
                print(f"Warning: Method {method} failed: {e}")
        
        if not predictions:
            raise ValueError("All prediction methods failed")
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict_with_confidence_threshold(self, image_path, confidence_threshold=0.3):
        """Predict with confidence thresholding"""
        ensemble_pred = self.predict_ensemble(image_path)
        
        # Get top predictions
        top_5_indices = np.argsort(ensemble_pred)[-5:][::-1]
        
        results = []
        for idx in top_5_indices:
            confidence = float(ensemble_pred[idx])
            if confidence >= confidence_threshold:
                results.append({
                    'class': self.class_names[idx],
                    'confidence': confidence * 100,
                    'index': idx
                })
        
        if not results:
            # If no predictions meet threshold, return top prediction with warning
            top_idx = top_5_indices[0]
            results.append({
                'class': self.class_names[top_idx],
                'confidence': float(ensemble_pred[top_idx]) * 100,
                'index': top_idx,
                'warning': 'Low confidence prediction'
            })
        
        return results
    
    def predict_with_uncertainty(self, image_path, num_samples=10):
        """Predict with uncertainty estimation using multiple samples"""
        predictions = []
        
        for _ in range(num_samples):
            # Add small random noise to input
            processed_img = self.preprocessor.preprocess_single_image(image_path, method='standard')
            
            # Add noise
            noise = np.random.normal(0, 0.01, processed_img.shape)
            noisy_img = processed_img + noise
            noisy_img = np.clip(noisy_img, 0, 1)
            
            pred = self.model.predict(noisy_img, verbose=0)
            predictions.append(pred[0])
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Get top prediction with uncertainty
        top_idx = np.argmax(mean_pred)
        
        return {
            'class': self.class_names[top_idx],
            'confidence': float(mean_pred[top_idx]) * 100,
            'uncertainty': float(std_pred[top_idx]) * 100,
            'reliable': std_pred[top_idx] < 0.1  # Low uncertainty = more reliable
        }
    
    def predict_robust(self, image_path):
        """Most robust prediction combining multiple techniques"""
        print(f"Starting robust prediction for {image_path}")
        try:
            # Method 1: Ensemble prediction
            ensemble_results = self.predict_with_confidence_threshold(image_path, confidence_threshold=0.2)
            
            # Method 2: Uncertainty estimation
            uncertainty_result = self.predict_with_uncertainty(image_path)
            
            # Method 3: Multiple preprocessing methods
            methods = ['standard', 'enhanced', 'contrast_enhanced']
            method_predictions = []
            
            for method in methods:
                try:
                    pred = self.predict_single_method(image_path, method=method)
                    top_idx = np.argmax(pred)
                    method_predictions.append({
                        'method': method,
                        'class': self.class_names[top_idx],
                        'confidence': float(pred[top_idx]) * 100
                    })
                except:
                    continue
            
            # Combine results
            result = {
                'primary_prediction': ensemble_results[0],
                'alternative_predictions': ensemble_results[1:3] if len(ensemble_results) > 1 else [],
                'uncertainty_analysis': uncertainty_result,
                'method_comparison': method_predictions,
                'recommendation': self._generate_recommendation(ensemble_results, uncertainty_result, method_predictions)
            }

            print(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in robust prediction: {str(e)}")
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _generate_recommendation(self, ensemble_results, uncertainty_result, method_predictions):
        """Generate recommendation based on all analyses"""
        primary = ensemble_results[0]
        
        # Check if all methods agree
        method_classes = [pred['class'] for pred in method_predictions]
        class_counts = Counter(method_classes)
        most_common_class = class_counts.most_common(1)[0]
        
        recommendation = {
            'confidence_level': 'high' if primary['confidence'] > 70 else 'medium' if primary['confidence'] > 40 else 'low',
            'reliability': 'high' if uncertainty_result.get('reliable', False) else 'medium',
            'consensus': f"{most_common_class[1]}/{len(method_predictions)} methods agree on '{most_common_class[0]}'"
        }
        
        if primary['confidence'] > 80 and uncertainty_result.get('reliable', False):
            recommendation['verdict'] = f"High confidence: {primary['class']}"
        elif primary['confidence'] > 50:
            recommendation['verdict'] = f"Moderate confidence: {primary['class']}"
        else:
            recommendation['verdict'] = f"Low confidence prediction. Consider retaking photo or checking if food item is in training data."
        
        return recommendation
