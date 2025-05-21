from catdogclassifier import SVMPredictor
image_path = "/content/downloaded_image.jpg"
if not os.path.exists(image_path):
  print(f"Image not found: {image_path}")
else:
  predictor = SVMPredictor()
  label = predictor.predict(image_path)
  print(f"\n🖼️ Predicted Label: {label}")
