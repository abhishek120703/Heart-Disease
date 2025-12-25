# web_app/app.py
"""
Flask Web App for AI-Driven Crop Disease Prediction & Management System.

Features:
- Upload an image
- Run prediction using trained PyTorch model
- Display predicted disease & confidence
"""

import sys
import os

# Add project root directory to Python PATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from pathlib import Path
from flask import Flask, render_template, request

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.predict import load_model, predict_image
from src.model import get_device


# ------------------------------------------------------------
# Flask Configuration
# ------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(Path(__file__).parent / "static" / "uploads")
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

# Paths
DATASET_PATH = ROOT / "data" / "plant_village"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"

# Load classes
CLASSES = json.load(open("classes.json"))

# Load model once
print("[INFO] Loading model for Flask app...")
MODEL, DEVICE = load_model(
    checkpoint_path=str(CHECKPOINT_PATH),
    classes=CLASSES,
    backbone="resnet18",
)
print("[INFO] Flask model loaded.")


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", error="No image uploaded!")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", error="No file selected!")

    # Save uploaded file
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    # Run prediction
    pred_class, confidence = predict_image(
        img_path=save_path,
        model=MODEL,
        classes=CLASSES,
        device=DEVICE,
    )

    return render_template(
        "index.html",
        prediction=pred_class,
        confidence=f"{confidence*100:.2f}%",
        image_path=f"static/uploads/{file.filename}",
    )


@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon/icon.png')

from flask import send_from_directory
# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

