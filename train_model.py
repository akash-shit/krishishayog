"""
KrishiMitra Plant Disease Detection Model Training Script
This script trains a CNN model to detect diseases in plant leaves
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Configuration
class TrainingConfig:
    # Data paths
    DATASET_PATH = 'dataset'  # Your dataset folder
    MODEL_SAVE_PATH = 'models/plant_disease_model.h5'
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    
    # Training parameters
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Model parameters
    NUM_CLASSES = 25  # Adjust based on your dataset
    
    # Class names (update based on your dataset)
    CLASS_NAMES = [
        'Apple_scab', 'Apple_black_rot', 'Apple_cedar_apple_rust', 'Apple_healthy',
        'Corn_cercospora_leaf_spot', 'Corn_common_rust', 'Corn_northern_leaf_blight', 'Corn_healthy',
        'Potato_early_blight', 'Potato_late_blight', 'Potato_healthy',
        'Rice_brown_spot', 'Rice_leaf_blight', 'Rice_neck_blast', 'Rice_healthy',
        'Tomato_bacterial_spot', 'Tomato_early_blight', 'Tomato_late_blight', 'Tomato_leaf_mold',
        'Tomato_septoria_leaf_spot', 'Tomato_spider_mites', 'Tomato_target_spot',
        'Tomato_yellow_leaf_curl_virus', 'Tomato_mosaic_virus', 'Tomato_healthy'
    ]

class PlantDiseaseModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def setup_data_generators(self):
        """Setup data generators with augmentation"""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.config.DATASET_PATH,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        self.validation_generator = val_datagen.flow_from_directory(
            self.config.DATASET_PATH,
            target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.validation_generator.samples} validation images")
        print(f"Number of classes: {self.train_generator.num_classes}")
        
    def create_model(self):
        """Create CNN model using transfer learning with MobileNetV2"""
        # Load pre-trained MobileNetV2 model
        base_model = MobileNetV2(
            input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom classification layers
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.config.NUM_CLASSES, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
    def train_model(self):
        """Train the model"""
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Calculate steps
        steps_per_epoch = self.train_generator.samples // self.config.BATCH_SIZE
        validation_steps = self.validation_generator.samples // self.config.BATCH_SIZE
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train model
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.config.EPOCHS,
            validation_data=self.validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
    def fine_tune_model(self):
        """Fine-tune the model by unfreezing some layers"""
        print("Starting fine-tuning...")
        
        # Unfreeze the last few layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Continue training
        fine_tune_epochs = 20
        total_epochs = self.config.EPOCHS + fine_tune_epochs
        
        history_fine = self.model.fit(
            self.train_generator,
            epochs=total_epochs,
            initial_epoch=self.history.epoch[-1],
            validation_data=self.validation_generator,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )
        
        # Combine histories
        self.history.history['accuracy'].extend(history_fine.history['accuracy'])
        self.history.history['val_accuracy'].extend(history_fine.history['val_accuracy'])
        self.history.history['loss'].extend(history_fine.history['loss'])
        self.history.history['val_loss'].extend(history_fine.history['val_loss'])
        
    def evaluate_model(self):
        """Evaluate model performance"""
        # Evaluate on validation set
        val_loss, val_accuracy, val_top_k = self.model.evaluate(self.validation_generator)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Top-5 Accuracy: {val_top_k:.4f}")
        
        # Generate predictions for confusion matrix
        Y_pred = self.model.predict(self.validation_generator)
        y_pred = np.argmax(Y_pred, axis=1)
        
        # Get true labels
        y_true = self.validation_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.config.CLASS_NAMES))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.config.CLASS_NAMES,
                   yticklabels=self.config.CLASS_NAMES)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        self.model.save(self.config.MODEL_SAVE_PATH)
        print(f"Model saved to {self.config.MODEL_SAVE_PATH}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open('models/model_architecture.json', 'w') as json_file:
            json_file.write(model_json)
        
        # Save class names
        import json
        with open('models/class_names.json', 'w') as f:
            json.dump(self.config.CLASS_NAMES, f)

def prepare_dataset_structure():
    """
    Instructions for preparing your dataset:
    
    Create a folder structure like this:
    dataset/
    ├── Apple_scab/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Apple_black_rot/
    │   ├── image1.jpg
    │   └── ...
    ├── Corn_healthy/
    ├── Potato_early_blight/
    ├── Rice_brown_spot/
    ├── Tomato_bacterial_spot/
    └── ...
    
    Each folder should contain images of that specific class.
    """
    print("Dataset Preparation Instructions:")
    print("1. Create a 'dataset' folder in the project root")
    print("2. Create subfolders for each disease/healthy class")
    print("3. Place images in corresponding folders")
    print("4. Ensure images are in JPG/PNG format")
    print("5. Recommended: 500-2000 images per class for best results")
    print("6. Images will be automatically resized to 224x224 pixels")

def main():
    """Main training pipeline"""
    print("KrishiMitra Plant Disease Detection Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    config = TrainingConfig()
    if not os.path.exists(config.DATASET_PATH):
        print(f"Dataset folder '{config.DATASET_PATH}' not found!")
        prepare_dataset_structure()
        return
    
    # Initialize trainer
    trainer = PlantDiseaseModelTrainer(config)
    
    try:
        # Setup data
        print("Setting up data generators...")
        trainer.setup_data_generators()
        
        # Create model
        print("Creating model...")
        trainer.create_model()
        
        # Train model
        print("Starting training...")
        trainer.train_model()
        
        # Fine-tune model
        print("Fine-tuning model...")
        trainer.fine_tune_model()
        
        # Evaluate model
        print("Evaluating model...")
        trainer.evaluate_model()
        
        # Plot results
        print("Plotting training history...")
        trainer.plot_training_history()
        
        # Save model
        print("Saving model...")
        trainer.save_model()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
Usage Instructions:

1. Install required packages:
   pip install tensorflow pillow matplotlib scikit-learn seaborn

2. Prepare your dataset in the required folder structure

3. Run the training script:
   python train_model.py

4. The script will:
   - Load and preprocess your images
   - Create a CNN model using transfer learning
   - Train the model with data augmentation
   - Fine-tune the model for better performance
   - Evaluate and save the model
   - Generate performance plots and confusion matrix

5. The trained model will be saved as 'models/plant_disease_model.h5'

Dataset Sources:
- PlantVillage Dataset
- Kaggle Plant Disease Datasets
- Agricultural Research Institution Datasets
"""