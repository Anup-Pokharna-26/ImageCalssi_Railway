#!/usr/bin/env python3
"""
Development setup script to create dummy model and class names for testing
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

def create_dummy_model():
    """Create a dummy CNN model for development/testing"""
    print("Creating dummy model for development...")
    
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    
    # Create a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dense(80, activation='softmax')  # 80 classes for Indian food
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save('model/best_model.keras')
    print("✅ Dummy model saved to model/best_model.keras")
    
    return model

def create_dummy_class_names():
    """Create dummy class names file"""
    print("Creating dummy class names...")
    
    # Indian food class names (matches the actual model)
    class_names = [
        'adhirasam', 'aloo_gobi', 'aloo_matar', 'aloo_methi', 'aloo_shimla_mirch',
        'aloo_tikki', 'anarsa', 'ariselu', 'bandar_laddu', 'basundi',
        'bhatura', 'bhindi_masala', 'biryani', 'boondi', 'butter_chicken',
        'chak_hao_kheer', 'cham_cham', 'chana_masala', 'chapati', 'chhena_kheeri',
        'chicken_razala', 'chicken_tikka', 'chicken_tikka_masala', 'chikki', 'daal_baati_churma',
        'daal_puri', 'dal_makhani', 'dal_tadka', 'dharwad_pedha', 'doodhpak',
        'double_ka_meetha', 'dum_aloo', 'gajar_ka_halwa', 'gavvalu', 'ghevar',
        'gulab_jamun', 'imarti', 'jalebi', 'kachori', 'kadai_paneer',
        'kadhi_pakoda', 'kajjikaya', 'kakinada_khaja', 'kalakand', 'karela_bharta',
        'kofta', 'kuzhi_paniyaram', 'lassi', 'ledikeni', 'litti_chokha',
        'lyangcha', 'maach_jhol', 'makki_di_roti_sarson_da_saag', 'malapua', 'misi_roti',
        'misti_doi', 'modak', 'mysore_pak', 'naan', 'navrattan_korma',
        'palak_paneer', 'paneer_butter_masala', 'phirni', 'pithe', 'poha',
        'poornalu', 'pootharekulu', 'qubani_ka_meetha', 'rabri', 'rasgulla',
        'ras_malai', 'sandesh', 'shankarpali', 'sheera', 'sheer_korma',
        'shrikhand', 'sohan_halwa', 'sohan_papdi', 'sutar_feni', 'unni_appam'
    ]
    
    # Write class names to file
    with open('model/class_names.txt', 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"✅ Created class names file with {len(class_names)} classes")
    return class_names

def create_improved_dummy_model():
    """Create a more realistic dummy model using transfer learning"""
    print("Creating improved dummy model with transfer learning...")
    
    # Create base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Add custom layers
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(80, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Save the model
    model.save('model/best_model.keras')
    print("✅ Improved dummy model saved to model/best_model.keras")
    print(f"📊 Model parameters: {model.count_params():,}")
    
    return model

def setup_development_environment():
    """Complete development setup"""
    print("🚀 Setting up development environment...")
    print("=" * 50)
    
    try:
        # Create directories
        os.makedirs('model', exist_ok=True)
        os.makedirs('static/uploads', exist_ok=True)
        
        # Create dummy files
        class_names = create_dummy_class_names()
        model = create_improved_dummy_model()
        
        print("\n" + "=" * 50)
        print("✅ Development environment setup complete!")
        print(f"📁 Model directory: model/")
        print(f"🤖 Dummy model: model/best_model.keras")
        print(f"📋 Class names: model/class_names.txt ({len(class_names)} classes)")
        print(f"📤 Upload directory: static/uploads/")
        
        print("\n🔥 You can now run the Flask app:")
        print("   python app.py")
        print("\n💡 The app will use these dummy files for testing.")
        print("💡 For production, configure real Google Drive URLs in download_model.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = setup_development_environment()
    if success:
        print("\n🎉 Ready to start developing!")
    else:
        print("\n💥 Setup failed. Please check the errors above.")
