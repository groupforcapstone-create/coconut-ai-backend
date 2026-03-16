import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CLOUD CONFIGURATION ---
MODEL_PATH = "coconut_model_v2_ultra.h5" 
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# --- LOAD AI MODEL (Memory Efficient) ---
def load_model_file():
    try:
        # Using compile=False saves memory on Render's free tier
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            print(f"Error: {MODEL_PATH} not found in root directory.")
            return None
    except Exception as e:
        print(f"Model Load Error: {e}")
        return None

model = load_model_file()

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    """Verify that the AI server is online"""
    status = "Model Loaded" if model else "Model Not Found"
    return jsonify({"status": "Server is Live", "model_status": status}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "AI Model failed to load on server"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image sent"}), 400
    
    # Receive address string from Flutter
    address = request.form.get('address', 'Unknown Location')

    try:
        # Convert uploaded file to OpenCV format
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Pre-process image for AI
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # Run Prediction
        preds = model.predict(img_final, verbose=0)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100

        # Response payload (No database logic here for speed and security)
        return jsonify({
            "variety_name": "Not a Coconut" if label == "NotCoconut" else label,
            "confidence": round(confidence, 2),
            "address": address,
            "status": "success"
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- RENDER DEPLOYMENT SETTING ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
