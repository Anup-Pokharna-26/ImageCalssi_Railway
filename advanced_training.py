#!/usr/bin/env python3
"""
Advanced training pipeline with better augmentation and transfer learning
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4, EfficientNetB3, ResNet152V2, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
class AdvancedFoodClassifier:
    def __init__(self, num_classes=80, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def create_advanced_augmentation(self):
        """Create advanced data augmentation pipeline"""
        return ImageDataGenerator(
            # Geometric transformations
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            
            # Intensity transformations
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            
            # Preprocessing
            rescale=1./255,
            fill_mode='nearest',
            
            # Validation split
            validation_split=0.2
        )
    
    def build_model(self):
        """Build advanced model with transfer learning"""
        print("Building advanced model with EfficientNetB3...")
        
        # Load pre-trained EfficientNetB3
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers initially
        for layer in base_model.layers:
            layer.trainable = False
            
        self.model = model
        self.base_model = base_model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with advanced settings"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
    def get_callbacks(self, model_name='best_food_model.keras'):
        """Get training callbacks"""
        return [
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'model/{model_name}',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
    
    def train_phase_1(self, train_generator, validation_generator, epochs=15):
        """Phase 1: Train with frozen base model"""
        print("Phase 1: Training with frozen base model...")
        
        history1 = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=self.get_callbacks('phase1_model.keras'),
            verbose=1
        )
        return history1
    
    def train_phase_2(self, train_generator, validation_generator, epochs=20):
        """Phase 2: Fine-tune with unfrozen layers"""
        print("Phase 2: Fine-tuning with unfrozen layers...")
        
        # Unfreeze top layers of base model
        for layer in self.base_model.layers[-50:]:
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=1e-5)
        
        history2 = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=self.get_callbacks('final_model.keras'),
            verbose=1
        )
        return history2
    
    def train_complete_pipeline(self, data_path, batch_size=32):
        """Complete training pipeline"""
        print("Starting complete training pipeline...")
        
        # Create data generators
        datagen = self.create_advanced_augmentation()
        
        train_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            data_path,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Build and compile model
        self.build_model()
        self.compile_model()
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {validation_generator.samples}")
        print(f"Classes: {train_generator.num_classes}")
        
        # Phase 1: Frozen base model
        history1 = self.train_phase_1(train_generator, validation_generator)
        
        # Phase 2: Fine-tuning
        history2 = self.train_phase_2(train_generator, validation_generator)
        
        return history1, history2, train_generator.class_indices

def create_improved_model():
    """Create and return an improved model"""
    classifier = AdvancedFoodClassifier(num_classes=80)
    
    # If you have your data organized in folders, uncomment and use:
    # history1, history2, class_indices = classifier.train_complete_pipeline('path/to/your/data')
    
    # For now, just build the model architecture
    model = classifier.build_model()
    classifier.compile_model()
    
    print("Advanced model created successfully!")
    print("Model summary:")
    model.summary()
    
    return model, classifier

if __name__ == '__main__':
    model, classifier = create_improved_model()
    
    # Save the improved architecture
    model.save('model/improved_architecture.keras')
    print("Improved model architecture saved!")
