from catdogclassify import CatDogPredictor
predictor = CatDogPredictor()
img_path = "/content/downloaded_image.jpg"
print("Prediction:", predictor.predict(img_path))