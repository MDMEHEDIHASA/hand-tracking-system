# gesture_model_training.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os  # ➕ ADDED
import sys  # ➕ ADDED
import argparse  # ➕ ADDED

class GestureRecognitionModel:
    def __init__(self, num_classes=18, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_cnn_model(self):
        """Build CNN model for static gesture recognition"""
        # ➕ ADDED: Print model summary
        print("\n" + "="*70)
        print("BUILDING CUSTOM CNN MODEL")
        print("="*70)
        
        model = models.Sequential([
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            # Fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        self.model = model
        
        # ➕ ADDED: Print model architecture
        print("\nModel Architecture:")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")
        print("="*70)
        
        return model
    
    def build_transfer_learning_model(self):
        """Build model using transfer learning with MobileNetV2"""
        # ➕ ADDED: Print model info
        print("\n" + "="*70)
        print("BUILDING TRANSFER LEARNING MODEL (MobileNetV2)")
        print("="*70)
        
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        self.model = model
        
        # ➕ ADDED: Print model info
        print("\nModel Architecture:")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
        print("="*70)
        
        return model
    
    def build_lstm_model(self, sequence_length=30):
        """Build LSTM model for dynamic gesture recognition"""
        # ➕ ADDED: Print model info
        print("\n" + "="*70)
        print("BUILDING LSTM MODEL FOR DYNAMIC GESTURES")
        print("="*70)
        
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, 63)),  # 21 landmarks * 3 (x,y,z)
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # ➕ ADDED: Print model info
        print("\nModel Architecture:")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")
        print("="*70)
        
        return model
    
    def train_model(self, train_data, val_data, epochs=50, model_name='gesture_model'):  # ✏️ CHANGED: Added model_name parameter
        """Train the model"""
        # ➕ ADDED: Print training info
        print("\n" + "="*70)
        print(f"STARTING TRAINING: {model_name}")
        print("="*70)
        print(f"Epochs: {epochs}")
        print(f"Training samples: {train_data.samples}")
        print(f"Validation samples: {val_data.samples}")
        print(f"Batch size: {train_data.batch_size}")
        print("="*70 + "\n")
        
        # ✏️ CHANGED: Updated checkpoint filename to use model_name
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1  # ➕ ADDED: verbose output
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1  # ➕ ADDED: verbose output
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'best_{model_name}.h5',  # ✏️ CHANGED: Use model_name
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1  # ➕ ADDED: verbose output
            ),
            # ➕ ADDED: TensorBoard callback
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/{model_name}',
                histogram_freq=1
            )
        ]
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1  # ✏️ CHANGED: Ensure verbose output
        )
        
        # ➕ ADDED: Save final model
        final_model_path = f'final_{model_name}.h5'
        self.model.save(final_model_path)
        print(f"\n✅ Model saved to: {final_model_path}")
        
        return history
    
    def evaluate_model(self, test_data, class_names):
        """Evaluate model and generate metrics"""
        # ➕ ADDED: Print evaluation info
        print("\n" + "="*70)
        print("EVALUATING MODEL")
        print("="*70)
        print(f"Test samples: {test_data.samples}")
        print("Generating predictions...")
        
        # Predictions
        y_pred = self.model.predict(test_data, verbose=1)  # ✏️ CHANGED: Added verbose
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_data.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, 
                                   target_names=class_names))
        
        # ➕ ADDED: Save classification report to file
        report = classification_report(y_true, y_pred_classes, 
                                      target_names=class_names)
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        print("✅ Classification report saved to: classification_report.txt")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        print("✅ Confusion matrix saved to: confusion_matrix.png")
        plt.show()
        
        # Calculate per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(12, 6))
        plt.bar(class_names, class_accuracy)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.tight_layout()
        plt.savefig('per_class_accuracy.png', dpi=300)
        print("✅ Per-class accuracy saved to: per_class_accuracy.png")
        plt.show()
        
        # ➕ ADDED: Calculate and display additional metrics
        overall_accuracy = np.mean(y_pred_classes == y_true)
        print(f"\n{'='*70}")
        print(f"OVERALL ACCURACY: {overall_accuracy*100:.2f}%")
        print(f"{'='*70}")
        
        return {
            'overall_accuracy': overall_accuracy,
            'per_class_accuracy': dict(zip(class_names, class_accuracy))
        }
    
    def plot_training_history(self, history, model_name='model'):  # ✏️ CHANGED: Added model_name
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title(f'{model_name} - Model Accuracy')  # ✏️ CHANGED: Added model name
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title(f'{model_name} - Model Loss')  # ✏️ CHANGED: Added model name
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        filename = f'training_history_{model_name}.png'  # ✏️ CHANGED: Use model name
        plt.savefig(filename, dpi=300)
        print(f"✅ Training history saved to: {filename}")
        plt.show()

# ➕ ADDED: Helper function to check dataset
def check_dataset_structure(dataset_path):
    """Check if dataset structure is correct"""
    print("\n" + "="*70)
    print("CHECKING DATASET STRUCTURE")
    print("="*70)
    
    required_dirs = ['train', 'val', 'test']
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return False
        
        # Count classes
        classes = [d for d in os.listdir(dir_path) 
                  if os.path.isdir(os.path.join(dir_path, d))]
        
        if len(classes) == 0:
            print(f"❌ No class folders found in: {dir_path}")
            return False
        
        # Count images per class
        total_images = 0
        for class_name in classes:
            class_path = os.path.join(dir_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(images)
        
        print(f"✅ {dir_name:10s}: {len(classes):2d} classes, {total_images:5d} images")
    
    print("="*70)
    return True

# ✏️ CHANGED: Complete rewrite with argument parsing and error handling
def train_gesture_model(dataset_path='dataset', model_type='cnn', epochs=50, skip_if_exists=False):
    """Complete training pipeline"""
    
    # ➕ ADDED: Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset path '{dataset_path}' does not exist!")
        print("\nExpected structure:")
        print("dataset/")
        print("  ├── train/")
        print("  │   ├── gesture1/")
        print("  │   ├── gesture2/")
        print("  │   └── ...")
        print("  ├── val/")
        print("  │   └── ...")
        print("  └── test/")
        print("      └── ...")
        return None
    
    # ➕ ADDED: Check dataset structure
    if not check_dataset_structure(dataset_path):
        print("\n❌ ERROR: Dataset structure is incorrect!")
        return None
    
    # ➕ ADDED: Check if model already exists
    model_filename = f'final_{model_type}_gesture_model.h5'
    if skip_if_exists and os.path.exists(model_filename):
        print(f"\n✅ Model '{model_filename}' already exists. Skipping training.")
        print("   Use --no-skip-if-exists to retrain.")
        return None
    
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # ➕ ADDED: Try-except for data loading
    try:
        # Load data
        train_generator = train_datagen.flow_from_directory(
            os.path.join(dataset_path, 'train'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            os.path.join(dataset_path, 'val'),
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        return None
    
    # ➕ ADDED: Display class information
    print("\n" + "="*70)
    print("CLASS INFORMATION")
    print("="*70)
    class_names = list(train_generator.class_indices.keys())
    print(f"Number of classes: {len(class_names)}")
    print("\nClasses:")
    for idx, name in enumerate(class_names, 1):
        print(f"  {idx:2d}. {name}")
    print("="*70)
    
    # Initialize model
    gesture_model = GestureRecognitionModel(num_classes=len(train_generator.class_indices))
    
    # ✏️ CHANGED: Conditional model building based on model_type
    if model_type == 'cnn':
        print("\n📊 Training Custom CNN Model...")
        gesture_model.build_cnn_model()
        history = gesture_model.train_model(train_generator, val_generator, 
                                          epochs=epochs, model_name='cnn_gesture_model')
        gesture_model.plot_training_history(history, 'CNN')
        
    elif model_type == 'mobilenet':
        print("\n📊 Training Transfer Learning Model (MobileNetV2)...")
        gesture_model.build_transfer_learning_model()
        history = gesture_model.train_model(train_generator, val_generator, 
                                          epochs=epochs, model_name='mobilenet_gesture_model')
        gesture_model.plot_training_history(history, 'MobileNetV2')
        
    elif model_type == 'both':
        # Train both models
        print("\n📊 Training BOTH models...")
        
        # CNN
        print("\n1️⃣ Training Custom CNN...")
        gesture_model.build_cnn_model()
        history_cnn = gesture_model.train_model(train_generator, val_generator, 
                                               epochs=epochs, model_name='cnn_gesture_model')
        gesture_model.plot_training_history(history_cnn, 'CNN')
        
        # MobileNet
        print("\n2️⃣ Training MobileNetV2...")
        gesture_model.build_transfer_learning_model()
        history_tl = gesture_model.train_model(train_generator, val_generator, 
                                              epochs=epochs, model_name='mobilenet_gesture_model')
        gesture_model.plot_training_history(history_tl, 'MobileNetV2')
        
        history = history_tl  # Use last one for evaluation
    else:
        print(f"❌ ERROR: Unknown model type '{model_type}'")
        print("   Valid options: 'cnn', 'mobilenet', 'both'")
        return None
    
    # Evaluate
    print("\n📈 Evaluating on test set...")
    test_generator = val_datagen.flow_from_directory(
        os.path.join(dataset_path, 'test'),
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    metrics = gesture_model.evaluate_model(test_generator, class_names)
    
    print(f"\n{'='*70}")
    print(f"FINAL ACCURACY: {metrics['overall_accuracy']*100:.2f}%")
    print(f"{'='*70}")
    
    return gesture_model, history, metrics

# ➕ ADDED: Main function with argument parsing
def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(description='Train Hand Gesture Recognition Model')
    
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Path to dataset directory (default: dataset)')
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'mobilenet', 'both'],
                       help='Model type to train (default: cnn)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--no-skip-if-exists', action='store_true',
                       help='Train even if model file already exists')
    
    args = parser.parse_args()
    
    # ➕ ADDED: Print configuration
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Dataset path: {args.dataset}")
    print(f"Model type: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Skip if exists: {not args.no_skip_if_exists}")
    print("="*70)
    
    # Train model
    result = train_gesture_model(
        dataset_path=args.dataset,
        model_type=args.model,
        epochs=args.epochs,
        skip_if_exists=not args.no_skip_if_exists
    )
    
    if result:
        print("\n✅ Training completed successfully!")
    else:
        print("\n❌ Training failed or was skipped!")
        sys.exit(1)

# ✏️ CHANGED: Use main function
if __name__ == "__main__":
    main()