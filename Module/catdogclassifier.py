import os
import cv2
import numpy as np
import joblib
from preprocessor import SVMPreprocessor  # Ensure this module exists and contains the class

class SVMPredictor:
    def __init__(self, model_dir="model_artifacts"):
        # Load preprocessor and pipeline
        self.preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
        self.pipeline = joblib.load(os.path.join(model_dir, "svm_pipeline.pkl"))
        self.image_size = self.preprocessor.image_size

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img

    def predict(self, image_path):
        img = self.load_image(image_path)
        features = self.preprocessor.transform_images([img])
        prediction = self.pipeline.predict(features)[0]
        return "Cat" if prediction == 0 else "Dog"