# Plant Disease Detection Website

This project is a comprehensive web application designed to assist farmers and agricultural professionals in identifying plant diseases with ease and accuracy. By leveraging advanced machine learning models, the application can detect and classify diseases in various plants, providing valuable insights for better crop management and decision-making.

## Features

- **Disease Detection**: Upload images of plants to receive real-time diagnosis of plant diseases. The system uses state-of-the-art AI models to ensure high accuracy and reliability.
- **Multi-Plant Support**: Currently supports disease detection for potatoes, corn, apples, and grapes, with plans to expand support to additional plant species in the future.
- **User-Friendly Interface**: A modern, green-themed design tailored for ease of use by farmers, featuring intuitive navigation and clear instructions.
- **Responsive Design**: Fully functional across different devices and screen sizes, ensuring accessibility for users on mobile, tablet, and desktop platforms.
- **Informative Website**: Includes sections for Home, About, Information, and plant-specific details, all presented in a sleek, dark design that enhances readability and user engagement.
- **Feedback Mechanism**: Users can provide feedback on the accuracy of predictions, helping to improve the AI model over time.

## Technologies Used

- **Flask**: Serves as the backend web framework, handling requests and serving the web pages.
- **TensorFlow & Keras**: Utilized for building and deploying the machine learning models that power the disease detection capabilities.
- **HTML/CSS/JavaScript**: Employed for the frontend design, creating a visually appealing and interactive user interface.
- **Python**: Used for server-side logic, model operations, and integrating various components of the application.

## Installation

1. **Clone the Repository**: Clone the project repository to your local machine using the following command:
   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   ```

2. **Navigate to the Project Directory**: Change into the project directory:
   ```bash
   cd plant-disease-detection
   ```

3. **Set Up a Virtual Environment**: It is recommended to use a virtual environment to manage dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install Dependencies**: Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download Model Weights**: Ensure that the model weight files are correctly placed in the designated directory as specified in the configuration.

## Usage

- **Running the Application**: Start the Flask server to run the application locally:
  ```bash
  python3 app.py
  ```
  Access the application by navigating to `http://localhost:5000` in your web browser.

- **Uploading Images**: Use the diagnostic tool to upload images of plant leaves and receive a diagnosis.

- **Providing Feedback**: After receiving a diagnosis, users can provide feedback on the accuracy of the prediction, which is valuable for improving the model.

## Architecture Overview

The application is structured into several key components:

- **Frontend**: Built with HTML, CSS, and JavaScript, providing a responsive and interactive user interface.
- **Backend**: Powered by Flask, handling HTTP requests, and serving the frontend assets.
- **Machine Learning Models**: Developed using TensorFlow and Keras, these models are responsible for analyzing images and predicting plant diseases.

## Contributing

Contributions are welcome! If you have ideas for improvements or have found bugs, please fork the repository and submit a pull request. We appreciate your help in making this project better.


