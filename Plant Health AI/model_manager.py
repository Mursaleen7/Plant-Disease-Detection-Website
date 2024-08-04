import tensorflow as tf
from config import AppConfig

class ModelManager:
    def __init__(self, model_weight_path, class_names):
        self.model_weight_path = model_weight_path
        self.class_names = class_names
        self.model = self.loadModel()

    def loadModel(self):
        num_classes = len(self.class_names)
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
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.build(input_shape=(None, AppConfig.IMAGE_SIZE, AppConfig.IMAGE_SIZE, 3))
        model.load_weights(self.model_weight_path)
        return model

    def predict(self, img_array):
        return self.model.predict(img_array)
