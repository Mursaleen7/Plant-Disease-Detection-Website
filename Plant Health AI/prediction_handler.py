import tempfile
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from config import AppConfig

class PredictionHandler:
    class_name_mapping = {
        'Potato___Early_blight': 'Early Blight Potato',
        'Potato___Late_blight': 'Late Blight Potato',
        'Potato___healthy': 'Healthy Potato',
        'Apple___Apple_scab': 'Apple Scab',
        'Apple___Black_rot': 'Apple Black Rot',
        'Apple___Cedar_apple_rust': 'Apple Cedar Apple Rust',
        'Apple___healthy': 'Healthy Apple',
        'Corn/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora Leaf Spot Gray Leaf Spot Corn',
        'Corn/Corn_(maize)___Common_rust_': 'Common Rust Corn',
        'Corn/Corn_(maize)___healthy': 'Healthy Corn',
        'Corn/Corn_(maize)___Northern_Leaf_Blight': 'Northern Leaf Blight Corn',
        'Grape/Grape___Black_rot': 'Black Rot Grape',
        'Grape/Grape___Esca_(Black_Measles)': 'Black Measles Grape',
        'Grape/Grape___healthy': 'Healthy Grape',
        'Grape/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight Grape',
    }

    @staticmethod
    def formatClassName(predicted_class):
        return PredictionHandler.class_name_mapping.get(predicted_class, predicted_class)

    @staticmethod
    def handlePrediction(file, plant_type, model_manager):
        temp_file_path = None
        
        try:
            # Save file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_file_path = temp_file.name

            # Load and preprocess the image
            try:
                img = image.load_img(temp_file_path, target_size=(AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, 0)  # Create batch axis
            except Exception as e:
                raise ValueError(f"Error loading or processing image: {str(e)}")

            # Predict
            try:
                predictions = model_manager.predict(img_array)
                if not isinstance(predictions, np.ndarray):
                    raise ValueError("Predictions are not in the expected format")

                if predictions.shape[0] == 0 or predictions.shape[1] == 0:
                    raise ValueError("Predictions array is empty")

                predicted_class = model_manager.class_names[np.argmax(predictions[0])]
            except Exception as e:
                raise ValueError(f"Error during prediction: {str(e)}")

            # Format the predicted class name
            formatted_class = PredictionHandler.formatClassName(predicted_class)
            confidence = round(100 * np.max(predictions[0]), 2)

            return {'predicted_class': formatted_class, 'confidence': confidence}

        except Exception as e:
            print(f'Error: {str(e)}')  # Print error to console
            return {'error': f'Prediction failed: {str(e)}'}

        finally:
            # Clean up
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

