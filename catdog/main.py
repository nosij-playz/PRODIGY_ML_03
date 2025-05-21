from catdogclassifier import SVMPredictor,SVMPreprocessor
import os
if __name__ == "__main__":
    image_path = "downloaded_image.jpg"  # Replace with your test image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
    else:
        predictor = SVMPredictor()
        label = predictor.predict(image_path)
        print(f"\n🖼️ Predicted Label: {label}")
