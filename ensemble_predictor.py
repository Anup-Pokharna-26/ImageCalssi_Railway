#!/usr/bin/env python3
"""
Ensemble prediction system for improved accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from enhanced_preprocessing import EnhancedPreprocessor
from collections import Counter

class EnsemblePredictor:
    def __init__(self, model_path, class_names_path):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
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
        print(f"🔍 Starting robust prediction for {image_path}")
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

            print(f"🔍 Prediction result: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Error in robust prediction: {str(e)}")
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

def test_ensemble_predictor():
    """Test the ensemble predictor"""
    MODEL_PATH = 'model/best_model.keras'
    CLASS_NAMES_PATH = 'model/class_names.txt'
    
    predictor = EnsemblePredictor(MODEL_PATH, CLASS_NAMES_PATH)
    
    # Test images
    test_images = [
        'static/uploads/biryani1.jpg',
        'static/uploads/biryani2.jpg',
        'static/uploads/aaluglobi.jpg'
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\n{'='*50}")
            print(f"Testing: {image_path}")
            print('='*50)
            
            result = predictor.predict_robust(image_path)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Print results
            primary = result['primary_prediction']
            print(f"🏆 Primary Prediction: {primary['class']} ({primary['confidence']:.2f}%)")
            
            if result['alternative_predictions']:
                print("📋 Alternative Predictions:")
                for i, alt in enumerate(result['alternative_predictions'], 2):
                    print(f"   {i}. {alt['class']} ({alt['confidence']:.2f}%)")
            
            uncertainty = result['uncertainty_analysis']
            print(f"🎯 Uncertainty Analysis: {uncertainty['class']} ({uncertainty['confidence']:.2f}% ± {uncertainty['uncertainty']:.2f}%)")
            print(f"   Reliable: {'Yes' if uncertainty['reliable'] else 'No'}")
            
            print("🔬 Method Comparison:")
            for method_pred in result['method_comparison']:
                print(f"   {method_pred['method']}: {method_pred['class']} ({method_pred['confidence']:.2f}%)")
            
            rec = result['recommendation']
            print(f"💡 Recommendation:")
            print(f"   Confidence: {rec['confidence_level']}")
            print(f"   Reliability: {rec['reliability']}")
            print(f"   Consensus: {rec['consensus']}")
            print(f"   Verdict: {rec['verdict']}")
        else:
            print(f"❌ Image not found: {image_path}")

if __name__ == '__main__':
    test_ensemble_predictor()
