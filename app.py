from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import requests
import numpy as np
from PIL import Image
import tensorflow as tf

# =====================
# APP CONFIG
# =====================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_cache")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

CLASS_LABELS = ["benign", "malignant", "normal"]
ALLOWED_EXT = {"jpg", "jpeg", "png"}

# =====================
# MODEL CONFIG (H5)
# =====================
MODEL_URL = "https://huggingface.co/mani880740255/skin_care_tflite/resolve/main/skin_cancer_mobilenetv2.h5"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_mobilenetv2.h5")

model = None

# =====================
# CHAT DATA
# =====================
CHAT_RESPONSES = {
    "what is skin care?": "Skin care is the practice of maintaining healthy, clean, and protected skin.",
    "what is a benign lesion?": "A benign lesion is non-cancerous and does not spread.",
    "what is a malignant lesion?": "A malignant lesion is cancerous and can spread.",
    "signs of skin cancer": "Irregular shape, color change, bleeding, rapid growth.",
    "how to prevent skin cancer?": "Use sunscreen, avoid excess sun, wear protective clothing."
}

# =====================
# HELPERS
# =====================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_model():
    global model
    if model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        print("Downloading H5 model...")
        r = requests.get(MODEL_URL, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Model download failed")

        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)

    print("Loading TensorFlow model...")
    model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(path):
    load_model()
    img = preprocess_image(path)
    preds = model.predict(img)[0]
    idx = int(np.argmax(preds))
    return idx, preds.tolist()

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Image required"}), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    try:
        idx, probs = predict_image(path)
        return jsonify({
            "model": "MobileNetV2 (H5)",
            "prediction": CLASS_LABELS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                CLASS_LABELS[i]: float(probs[i]) for i in range(3)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").lower().strip()

    if not msg:
        return jsonify({"suggestions": list(CHAT_RESPONSES.keys())[:3]})

    if msg in CHAT_RESPONSES:
        return jsonify({"reply": CHAT_RESPONSES[msg], "suggestions": []})

    return jsonify({
        "reply": "I only answer basic skin health questions.",
        "suggestions": list(CHAT_RESPONSES.keys())
    })

# =====================
# ENTRY
# =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

