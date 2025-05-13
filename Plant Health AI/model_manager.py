import tensorflow as tf
import numpy as np
import os
import json
import time
from datetime import datetime
from config import AppConfig

class ModelManager:
    def __init__(self, model_weight_path, class_names):
        self.model_weight_path = model_weight_path
        self.class_names = class_names
        self.model = self.loadModel()
        self.model_stats = {
            'last_loaded': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'predictions_made': 0,
            'avg_inference_time': 0,
            'total_inference_time': 0
        }
        # Initialize model metrics
        self.prediction_history = []
        self.prediction_distribution = {class_name: 0 for class_name in class_names}
        self.confidence_stats = {'min': 1.0, 'max': 0.0, 'avg': 0.0, 'total': 0.0, 'count': 0}
        
        # Log model initialization
        print(f"Model loaded from {model_weight_path} with {len(class_names)} classes")
        
    def loadModel(self):
        """Load the model with specified architecture and weights."""
        print(f"Loading model from {self.model_weight_path}...")
        num_classes = len(self.class_names)
        
        # Check if mobilenet version is available for transfer learning
        use_transfer_learning = AppConfig.USE_TRANSFER_LEARNING if hasattr(AppConfig, 'USE_TRANSFER_LEARNING') else False
        
        if use_transfer_learning:
            print("Using MobileNetV2 transfer learning model")
            return self._loadTransferLearningModel(num_classes)
        else:
            print("Using custom CNN architecture")
            return self._loadCustomCNNModel(num_classes)
    
    def _loadCustomCNNModel(self, num_classes):
        """Load a custom CNN model architecture."""
        model = tf.keras.Sequential([
            tf.keras.layers.Resizing(AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.build(input_shape=(None, AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE, 3))
        model.load_weights(self.model_weight_path)
        return model
    
    def _loadTransferLearningModel(self, num_classes):
        """Load a transfer learning model based on MobileNetV2."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            tf.keras.layers.Resizing(AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE),
            tf.keras.layers.Rescaling(1./127.5, offset=-1),  # MobileNet normalization
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.load_weights(self.model_weight_path)
        return model

    def predict(self, img_array):
        """Make a prediction with timing and statistics updates."""
        # Start timing
        start_time = time.time()
        
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Update statistics
        self.model_stats['predictions_made'] += 1
        self.model_stats['total_inference_time'] += inference_time
        self.model_stats['avg_inference_time'] = (
            self.model_stats['total_inference_time'] / self.model_stats['predictions_made']
        )
        
        # Record prediction for analytics
        self._recordPrediction(predictions[0])
        
        return predictions
    
    def _recordPrediction(self, prediction):
        """Record prediction for model analytics."""
        predicted_class_idx = np.argmax(prediction)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(np.max(prediction))
        
        # Update prediction distribution
        self.prediction_distribution[predicted_class] += 1
        
        # Update confidence statistics
        self.confidence_stats['min'] = min(self.confidence_stats['min'], confidence)
        self.confidence_stats['max'] = max(self.confidence_stats['max'], confidence)
        self.confidence_stats['total'] += confidence
        self.confidence_stats['count'] += 1
        self.confidence_stats['avg'] = self.confidence_stats['total'] / self.confidence_stats['count']
        
        # Add to history (capped at 100 entries)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.prediction_history.append({
            'timestamp': timestamp,
            'class': predicted_class,
            'confidence': confidence
        })
        
        # Keep history at a manageable size
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
    
    def getModelStats(self):
        """Get current model statistics."""
        return {
            'model_info': {
                'weight_path': self.model_weight_path,
                'class_count': len(self.class_names),
                'classes': self.class_names
            },
            'usage_stats': self.model_stats,
            'prediction_distribution': self.prediction_distribution,
            'confidence_stats': self.confidence_stats
        }
    
    def exportModelStats(self, filepath):
        """Export model statistics to a JSON file."""
        stats = self.getModelStats()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(stats, f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting model stats: {str(e)}")
            return False
    
    def evaluateModel(self, validation_data, validation_labels):
        """Evaluate model performance on validation data."""
        if not isinstance(validation_data, np.ndarray) or not isinstance(validation_labels, np.ndarray):
            raise ValueError("Validation data and labels must be numpy arrays")
        
        # Get model predictions
        predictions = self.model.predict(validation_data)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate accuracy
        true_classes = np.argmax(validation_labels, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Calculate class-wise accuracy
        class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            class_indices = np.where(true_classes == i)[0]
            if len(class_indices) > 0:
                class_acc = np.mean(predicted_classes[class_indices] == i)
                class_accuracy[class_name] = float(class_acc)
            else:
                class_accuracy[class_name] = 0.0
        
        return {
            'accuracy': float(accuracy),
            'class_accuracy': class_accuracy,
            'sample_count': len(validation_data)
        }
