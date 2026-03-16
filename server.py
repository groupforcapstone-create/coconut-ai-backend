import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = "coconut_model_v2_ultra.h5" 
IMG_SIZE = (224, 224)
# Ensure these match your Laravel 'detections' table variety names exactly
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# --- LOAD AI MODEL ---
def load_model_file():
    try:
        if os.path.exists(MODEL_PATH):
            # compile=False is critical for saving RAM on Render
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Critical Error: {MODEL_PATH} not found.")
        return None
    except Exception as e:
        print(f"Model Load Error: {e}")
        return None

model = load_model_file()

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    status = "Online" if model else "Model Error"
    return jsonify({"status": "AI Server Live", "model": status}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "AI Model is not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Metadata passed from Flutter
    address = request.form.get('address', 'Unknown Location')

    try:
        # Convert file to OpenCV
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Pre-process
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # Run Prediction
        preds = model.predict(img_final, verbose=0)[0]
        idx = np.argmax(preds)
        
        # 1. Create the Confidence Breakdown (for Laravel's confidence_json)
        confidence_json = []
        for i, name in enumerate(CLASS_NAMES):
            confidence_json.append({
                "label": "Not a Coconut" if name == "NotCoconut" else name,
                "confidence": round(float(preds[i]) * 100, 2)
            })

        # Sort by highest confidence first
        confidence_json = sorted(confidence_json, key=lambda x: x['confidence'], reverse=True)

        # 2. Identify top result
        label = CLASS_NAMES[idx]
        top_confidence = round(float(preds[idx]) * 100, 2)

        # 3. Return to Flutter
        return jsonify({
            "status": "success",
            "variety_name": "Not a Coconut" if label == "NotCoconut" else label,
            "confidence": top_confidence,
            "confidence_json": confidence_json, # Send this to Laravel storeScanHistory
            "address": address
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- DEPLOYMENT ---
if __name__ == "__main__":
    # Render requires dynamic port binding
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
