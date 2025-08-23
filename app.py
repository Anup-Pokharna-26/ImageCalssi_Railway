import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import time
from download_model import download_model, download_class_names, get_model_timestamp, COLAB_NOTEBOOK_URL
from ensemble_predictor_fixed import EnsemblePredictorFixed
from smart_preprocessing import SmartPreprocessor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'model/best_model.keras'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Global variables
model = None
CLASS_NAMES = []
last_model_check = 0
last_model_timestamp = 0

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load class names from file
def load_class_names():
    try:
        with open('model/class_names.txt', 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Class names file not found. Downloading...")
        try:
            return download_class_names()
        except Exception as e:
            print(f"Error downloading class names: {e}")
            # Return default class names as fallback
            return ["class_0", "class_1", "class_2", "class_3", "class_4"]

# Load the model
def load_model():
    global last_model_timestamp
    
    try:
        # Check if model exists, if not download it
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Downloading...")
            download_model()
        
        # Load the model with compile=False to avoid optimizer issues
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Recompile with appropriate settings
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        last_model_timestamp = get_model_timestamp()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with compile=False...")
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Model loaded with compile=False")
            print(f"Model input shape: {model.input_shape}")
            return model
        except Exception as e2:
            print(f"Still failed: {e2}")
            return None

# Check if model needs to be reloaded (every 5 minutes)
def check_model_reload():
    global model, last_model_check, last_model_timestamp
    
    current_time = time.time()
    
    # Check every 5 minutes
    if current_time - last_model_check > 300:  # 300 seconds = 5 minutes
        last_model_check = current_time
        current_timestamp = get_model_timestamp()
        
        # If timestamp changed, reload model
        if current_timestamp > last_model_timestamp:
            print("Model updated. Reloading...")
            model = load_model()
            # Also reload class names
            global CLASS_NAMES
            CLASS_NAMES = load_class_names()

# Initialize variables with safe defaults
CLASS_NAMES = []
model = None
ensemble_predictor = None
last_model_check = time.time()
last_model_timestamp = 0

def initialize_app():
    """Initialize app components with error handling"""
    global CLASS_NAMES, model, ensemble_predictor, last_model_timestamp
    
    print("🔄 Initializing application...")
    
    try:
        # Load class names first (faster and less likely to fail)
        CLASS_NAMES = load_class_names()
        print(f"✅ Loaded {len(CLASS_NAMES)} class names")
    except Exception as e:
        print(f"⚠️  Error loading class names: {e}")
        CLASS_NAMES = ["class_0", "class_1", "class_2", "class_3", "class_4"]
    
    # Try to load model (this might take time or fail)
    try:
        model = load_model()
        if model is not None:
            print("✅ Model loaded successfully")
        else:
            print("⚠️  Model loading failed, but app will continue")
    except Exception as e:
        print(f"⚠️  Error during model loading: {e}")
        model = None
    
    # Initialize ensemble predictor only if model is available
    if model is not None and CLASS_NAMES:
        try:
            ensemble_predictor = EnsemblePredictorFixed(model, CLASS_NAMES)
            print("✅ Enhanced ensemble predictor initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize ensemble predictor: {e}")
            ensemble_predictor = None
    
    print("🚀 Application initialization complete")

# Initialize the application
try:
    initialize_app()
except Exception as e:
    print(f"❌ Error during app initialization: {e}")
    print("🔄 App will continue with limited functionality")



def prepare_image(filepath, target_size=(224, 224)):
    """Preprocess image to RGB for current model"""
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)  # always 3 channels
    if img is None:
        raise ValueError("Could not read image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def index():
    try:
        return render_template('index.html', colab_url=COLAB_NOTEBOOK_URL)
    except Exception as e:
        print(f"Error serving index page: {e}")
        # Fallback response if template fails
        return f"""
        <html>
        <head><title>Image Classifier</title></head>
        <body>
            <h1>Image Classifier</h1>
            <p>Application is starting up... Please refresh in a moment.</p>
            <p>Model Status: {'Loaded' if model is not None else 'Loading...'}</p>
            <p>Classes Status: {len(CLASS_NAMES)} classes loaded</p>
        </body>
        </html>
        """, 200

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response

@app.route('/health')
def health_check():
    """Health check endpoint for Railway deployment"""
    # return jsonify({
    #     'status': 'healthy',
    #     'timestamp': time.time(),
    #     'model_loaded': model is not None,
    #     'classes_loaded': len(CLASS_NAMES) > 0
    # }), 200
    return "OK", 200

@app.route('/retrain')
def retrain():
    # Redirect to the Colab notebook for retraining
    return redirect(COLAB_NOTEBOOK_URL)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model needs to be reloaded
    check_model_reload()
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess the image
            try:
                img = prepare_image(filepath)
            except Exception as e:
                return jsonify({'error': f'Unable to read the image file: {str(e)}'}), 400
            
            # Make prediction
            prediction = model.predict(img)

            
            # Get top 3 predictions for better user experience
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            
            predictions = []
            for idx in top_3_indices:
                if idx < len(CLASS_NAMES):
                    predictions.append({
                        'class': CLASS_NAMES[idx],
                        'confidence': float(prediction[0][idx] * 100)
                    })
            
            # Main prediction (highest confidence)
            predicted_class_idx = top_3_indices[0]
            if predicted_class_idx >= len(CLASS_NAMES):
                predicted_class = f"Class_{predicted_class_idx}"
            else:
                predicted_class = CLASS_NAMES[predicted_class_idx]
            
            confidence = float(np.max(prediction) * 100)
            
            return jsonify({
                'class': predicted_class,
                'confidence': confidence,
                'image_path': f"/static/uploads/{filename}",
                'top_predictions': predictions,
                'debug_info': {
                    'total_classes': len(CLASS_NAMES),
                    'prediction_sum': float(prediction.sum()),
                    'model_confident': confidence > 50.0
                }
            })
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/predict_enhanced', methods=['POST'])
def predict_enhanced():
    """Enhanced prediction with ensemble methods and uncertainty estimation"""
    # Check if ensemble predictor is available
    if ensemble_predictor is None:
        return jsonify({'error': 'Enhanced predictor is not available. Using standard prediction.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Use ensemble prediction
            result = ensemble_predictor.predict_robust(filepath)
            
            if 'error' in result:
                return jsonify({'error': result['error']}), 500
            
            # Format response for frontend
            primary = result['primary_prediction']
            
            # Convert NumPy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()  # Convert NumPy scalars
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            # Convert all NumPy types in the result
            result = convert_numpy_types(result)
            primary = result['primary_prediction']
            
            response = {
                'method': 'enhanced_ensemble',
                'class': primary['class'],
                'confidence': float(primary['confidence']),
                'image_path': f"/static/uploads/{filename}",
                'enhanced_results': {
                    'primary_prediction': primary,
                    'alternative_predictions': result.get('alternative_predictions', []),
                    'uncertainty_analysis': result.get('uncertainty_analysis', {}),
                    'method_comparison': result.get('method_comparison', []),
                    'recommendation': result.get('recommendation', {})
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during enhanced prediction: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'})

@app.route('/predict_smart', methods=['POST'])
def predict_smart():
    """Super-smart prediction with advanced preprocessing and multiple strategies"""
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Use smart preprocessing
            smart_preprocessor = SmartPreprocessor()
            
            # Create multiple versions of the image
            versions = smart_preprocessor.create_multiple_versions(filepath)
            
            # Get predictions for all versions
            all_predictions = []
            for version in versions:
                pred = model.predict(np.expand_dims(version, axis=0), verbose=0)
                all_predictions.append(pred[0])
            
            # Ensemble the predictions
            ensemble_pred = np.mean(all_predictions, axis=0)
            
            # Get top 5 predictions
            top_5_indices = np.argsort(ensemble_pred)[-5:][::-1]
            
            predictions = []
            for idx in top_5_indices:
                if idx < len(CLASS_NAMES):
                    predictions.append({
                        'class': CLASS_NAMES[idx],
                        'confidence': float(ensemble_pred[idx] * 100)
                    })
            
            # Main prediction
            predicted_class = CLASS_NAMES[top_5_indices[0]]
            confidence = float(ensemble_pred[top_5_indices[0]] * 100)
            
            # Calculate consensus
            individual_preds = []
            for i, pred in enumerate(all_predictions):
                top_idx = np.argmax(pred)
                individual_preds.append({
                    'version': f'v{i+1}',
                    'class': CLASS_NAMES[top_idx],
                    'confidence': float(pred[top_idx] * 100)
                })
            
            # Check consensus
            predicted_classes = [p['class'] for p in individual_preds]
            consensus_class = max(set(predicted_classes), key=predicted_classes.count)
            consensus_count = predicted_classes.count(consensus_class)
            consensus_ratio = consensus_count / len(predicted_classes)
            
            response = {
                'method': 'smart_ensemble',
                'class': predicted_class,
                'confidence': confidence,
                'image_path': f"/static/uploads/{filename}",
                'top_predictions': predictions,
                'smart_analysis': {
                    'individual_predictions': individual_preds,
                    'consensus_class': consensus_class,
                    'consensus_ratio': consensus_ratio,
                    'reliability': 'high' if consensus_ratio >= 0.75 else 'medium' if consensus_ratio >= 0.5 else 'low',
                    'recommendation': (
                        f"High confidence: {predicted_class}" if confidence > 70 and consensus_ratio >= 0.75
                        else f"Moderate confidence: {predicted_class}" if confidence > 50
                        else "Low confidence - consider retaking photo with better lighting and focus on the food item"
                    )
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Error during smart prediction: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed'})




if __name__ == '__main__':
    # Production-ready configuration
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting application on port {port}")
    print(f"🔧 Debug mode: {debug_mode}")
    print(f"🤖 Model loaded: {model is not None}")
    print(f"📋 Classes loaded: {len(CLASS_NAMES)}")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)
