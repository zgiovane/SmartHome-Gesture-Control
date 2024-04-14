import cv2  # Importing the OpenCV library for image processing
import numpy as np  # Importing NumPy for numerical operations
import tensorflow as tf  # Importing TensorFlow for machine learning operations

keras = tf.keras  # Importing Keras for building and using neural networks
load_model = keras.models.load_model  # Function to load a pre-trained model
Model = keras.models.Model  # Class representing a Keras model

# Importing the os.path module for file path operations
import os.path  

# Define the base directory path
BASE = os.path.dirname(os.path.abspath(__file__))

class HandShapeFeatureExtractor:
    # Singleton class for extracting hand shape features using a pre-trained CNN model.

    __single = None  # Class variable to hold the single instance of the class

    @staticmethod
    def get_instance():
        # Static method to get the singleton instance of the class.
        if HandShapeFeatureExtractor.__single is None:
            HandShapeFeatureExtractor()
        return HandShapeFeatureExtractor.__single

    def __init__(self):
        # Constructor method to initialize the singleton instance with the pre-trained model.
        if HandShapeFeatureExtractor.__single is None:
            # Load the pre-trained CNN model
            real_model = load_model(os.path.join(BASE, 'gestures_trained_cnn_model.h5'))
            self.model = real_model
            HandShapeFeatureExtractor.__single = self

        else:
            raise Exception("This Class bears the model, so it is made Singleton")

    def __pre_process_input_image(crop):
        # Private method to preprocess the input image before feeding it to the model.
        try:
            # Resize the image to the expected input size of the model
            img = cv2.resize(crop, (300, 300))
            # Normalize the pixel values
            img_arr = np.array(img) / 255.0
            # Ensure the image has 3 channels if it's grayscale
            if len(img_arr.shape) == 2 or img_arr.shape[2] == 1:
                img_arr = np.stack((img_arr,) * 3, axis=-1)
            img_arr = img_arr.reshape(1, 300, 300, 3)  # Reshape for the model input
            return img_arr
        except Exception as e:
            print(str(e))
            raise

    def __bound_box(x, y, max_y, max_x):
        # Private method to calculate the bounding box for cropping specific hand parts.
        y1 = y + 80
        y2 = y - 80
        x1 = x + 80
        x2 = x - 80
        if max_y < y1:
            y1 = max_y
        if y - 80 < 0:
            y2 = 0
        if x + 80 > max_x:
            x1 = max_x
        if x - 80 < 0:
            x2 = 0
        return y1, y2, x1, x2

    def extract_feature(self, image):
        # Method to extract hand shape features from an image using the pre-trained model.
        try:
            img_arr = self.__pre_process_input_image(image)
            return self.model.predict(img_arr)
        except Exception as e:
