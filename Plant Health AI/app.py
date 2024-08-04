from flask import Flask, request, jsonify
from model_manager import ModelManager
from prediction_handler import PredictionHandler
from config import AppConfig, PLANT_MODELS
import tempfile
import os

# Initialize Flask app
app = AppConfig.initializeApp()

@app.route('/')
def index():
    return app.send_static_file('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the 'file' key is in request.files and 'plantType' is in request.form
    if 'file' not in request.files or 'plantType' not in request.form:
        return jsonify({'error': 'No file or plant type selected'}), 400

    file = request.files['file']
    plant_type = request.form['plantType']
    
    # Check if file is empty or has no filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate plant type
    if plant_type not in PLANT_MODELS:
        return jsonify({'error': 'Invalid plant type'}), 400

    # Load model and class names based on the selected plant type
    model_info = PLANT_MODELS[plant_type]
    
    # Validate model weight path
    if not os.path.isfile(model_info['model_weight_path']):
        return jsonify({'error': 'Model weight file not found'}), 500

    model_manager = ModelManager(model_weight_path=model_info['model_weight_path'], class_names=model_info['class_names'])

    try:
        # Ensure the file size is within acceptable limits
        max_file_size = 5 * 1024 * 1024  # 5 MB
        if file.content_length > max_file_size:
            return jsonify({'error': 'File size exceeds limit of 5MB'}), 400

        # Handle the prediction
        result = PredictionHandler.handlePrediction(file, plant_type, model_manager)

        # Check if the result contains an error
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)

    except Exception as e:
        print(f'Error: {str(e)}')  # Print error to console
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
