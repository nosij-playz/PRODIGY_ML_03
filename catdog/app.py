from flask import Flask, request, render_template, redirect, url_for
from catdogclassifier import SVMPredictor, SVMPreprocessor
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize predictor once
predictor = SVMPredictor()

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            try:
                label = predictor.predict(filepath)
            except Exception as e:
                label = f"Prediction error: {e}"
            os.remove(filepath)  # clean up
    return render_template("index.html", label=label)

if __name__ == "__main__":
    app.run(debug=True)
