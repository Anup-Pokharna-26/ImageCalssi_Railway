# Cloud-Based Image Classification System

A fully cloud-based AI web application for image classification that uses Google Drive for model storage and Google Colab for training.

## Features

- 🔮 Predict the class of uploaded images using a pre-trained deep learning model
- ☁️ Dynamically load the model from Google Drive
- 🔄 Automatic model reloading when updates are detected
- 🧠 Retrain the model through a Google Colab notebook
- 📊 Display prediction results with confidence scores

## Architecture

- **Model Training**: Google Colab + TensorFlow
- **Model Storage**: Google Drive
- **Web Framework**: Flask (Python)
- **Deployment**: Free cloud platforms (Render, Railway, Replit, etc.)
- **Image Handling**: OpenCV
- **Model Download**: gdown Python package

## Setup Instructions

### Prerequisites

1. A Google account with access to Google Drive and Google Colab
2. A trained model saved in Google Drive (in .keras format)
3. A text file with class names in Google Drive

### Configuration

1. Update the `download_model.py` file with your Google Drive URLs:
   - `MODEL_DRIVE_URL`: The shared link to your model file
   - `CLASS_NAMES_URL`: The shared link to your class names file
   - `COLAB_NOTEBOOK_URL`: The link to your Colab notebook for retraining

2. Make sure your Google Drive files are shared with appropriate permissions (anyone with the link can view)

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. **For Development/Testing** (if you don't have a trained model yet):

Run the development setup to create a dummy model:

```bash
python setup_dev.py
```

This will create:
- A dummy CNN model (`model/best_model.keras`)
- Sample class names (`model/class_names.txt`)

3. **For Production** (with your actual trained model):

Ensure your Google Drive URLs in `download_model.py` are correct and public.

4. Run the application:

```bash
python app.py
```

The application will be available at `http://localhost:8080`

## Troubleshooting

### Common Issues and Solutions

1. **"Server error: 500" when making predictions**
   - **Solution**: Run `python setup_dev.py` to create a dummy model for testing
   - **Cause**: Missing model file or incorrect Google Drive URLs

2. **"Model is not loaded" error**
   - **Solution**: Check that `model/best_model.keras` exists
   - **Alternative**: Run the development setup script

3. **Google Drive download fails**
   - **Solution**: Ensure your Google Drive URLs are in the correct format:
     - ❌ Wrong: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
     - ✅ Correct: `https://drive.google.com/uc?id=FILE_ID`
   - **Solution**: Make sure files are publicly accessible ("Anyone with the link can view")

4. **"Unable to read the image file" error**
   - **Solution**: Upload a valid image file (PNG, JPG, JPEG)
   - **Cause**: Corrupted or unsupported image format

### Development vs Production Setup

- **Development**: Use `setup_dev.py` to create dummy files for testing
- **Production**: Configure real Google Drive URLs in `download_model.py` ✅ **CURRENTLY ACTIVE**

### Current Model Status

✅ **Your trained model is now active!**
- **Model**: Indian Food Classification (80 classes)
- **Input Size**: 224x224 RGB images
- **Classes**: adhirasam, aloo_gobi, aloo_matar, biryani, butter_chicken, gulab_jamun, and 74 more Indian dishes
- **Parameters**: 2.4 million
- **Source**: Downloaded from your Google Drive

## File Structure

```
Cloud Pro/
├── app.py                 # Main Flask application
├── download_model.py      # Google Drive integration
├── setup_dev.py          # Development setup script
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── templates/
│   └── index.html        # Web interface
├── static/uploads/       # Uploaded images
└── model/
    ├── best_model.keras  # AI model file
    └── class_names.txt   # Class labels
```
