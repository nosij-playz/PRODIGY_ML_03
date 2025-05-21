import cv2
import joblib
import numpy as np

class CatDogPredictor:
    def __init__(self, preprocessor_path="preprocessor.pkl", model_path="svm_model.pkl"):
        self.preprocessor = joblib.load(preprocessor_path)
        self.model = joblib.load(model_path)

    def predict(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")
        img = cv2.resize(img, (self.preprocessor.image_size, self.preprocessor.image_size))
        features = self.preprocessor.transform_images([img])
        pred = self.model.predict(features)[0]
        return "Dog" if pred == 1 else "Cat"

