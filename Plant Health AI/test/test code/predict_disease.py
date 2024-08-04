import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from PIL import Image

# Constants
IMAGE_SIZE = 256
CHANNELS = 3
MODEL_WEIGHT_PATH = ''

# Load dataset (for class names)
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=32
)
class_names = dataset.class_names

# Load the model with the same architecture as in training
input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = len(class_names)

resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1./255),
])

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Load model weights
model.load_weights(MODEL_WEIGHT_PATH)
print("Model weights loaded.")

# Prediction function
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Load and preprocess an image for prediction
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
    return img

# Predict on a sample image
image_path = ""  # Update this path to your image file
img = load_and_preprocess_image(image_path)
predicted_class, confidence = predict(model, img)

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}%")

# Display the image and prediction
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(f"Predicted: {predicted_class} with Confidence: {confidence}%")
plt.axis("off")
plt.show()

