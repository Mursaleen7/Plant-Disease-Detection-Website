import os
from flask import Flask

class AppConfig:
    IMAGE_SIZE = 256

    @staticmethod
    def initializeApp():
        """Initialize and return a Flask application."""
        app = Flask(__name__)
        return app

def validate_model_path(path):
    """Check if the model weights file path is valid."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"Model weights file is empty: {path}")

def create_plant_models():
    """Create and return a dictionary of plant models with validated paths."""
    models = {
        'potato': {
            'class_names': ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
            'model_weight_path': '/Users/mursaleensakoskar/Desktop/Plant Health AI/weights/potato_model.weights.h5'
        },
        'apple': {
            'class_names': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
            'model_weight_path': '/Users/mursaleensakoskar/Desktop/Plant Health AI/weights/apple_model.weights.h5'
        },
        'corn': {
            'class_names': ['Corn/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn/Corn_(maize)___Common_rust_', 'Corn/Corn_(maize)___Northern_Leaf_Blight', 'Corn/Corn_(maize)___healthy'],
            'model_weight_path': '/Users/mursaleensakoskar/Desktop/Plant Health AI/weights/corn_model.weights.h5'
        },
        'grape': {
            'class_names': ['Grape/Grape___Black_rot', 'Grape/Grape___Esca_(Black_Measles)', 'Grape/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape/Grape___healthy'],
            'model_weight_path': '/Users/mursaleensakoskar/Desktop/Plant Health AI/weights/grape_model.weights.h5'
        }
    }

    # Validate paths for each plant model
    for plant, model_info in models.items():
        try:
            validate_model_path(model_info['model_weight_path'])
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: {e} for plant '{plant}'")

    return models

PLANT_MODELS = create_plant_models()
