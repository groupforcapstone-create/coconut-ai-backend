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

# IMPORTANT: Ensure these match EXACTLY with the 'variety_name' column in your Laravel 'detections' table
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# --- LOAD AI MODEL ---
def load_model_file():
    try:
        if os.path.exists(MODEL_PATH):
            # Using compile=False saves memory on Render's free tier
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            print(f"Error: {MODEL_PATH} not found.")
            return None
    except Exception as e:
        print(f"Model Load Error: {e}")
        return None

model = load_model_file()

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    status = "Model Loaded" if model else "Model Not Found"
    return jsonify({"status": "AI Server Live", "model_status": status}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "AI Model failed to load on server"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image sent"}), 400
    
    # Receive metadata from Flutter
    address = request.form.get('address', 'Unknown Location')

    try:
        # Convert file to OpenCV
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Pre-process image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # Run Prediction
        preds = model.predict(img_final, verbose=0)[0]
        idx = np.argmax(preds)
        
        # Prepare Confidence breakdown for Laravel's confidence_json
        # This builds a list of all classes and their scores
        all_confidences = []
        for i, class_name in enumerate(CLASS_NAMES):
            all_confidences.append({
                "label": class_name if class_name != "NotCoconut" else "Not a Coconut",
                "confidence": round(float(preds[i]) * 100, 2)
            })

        # Sort confidences so the highest is first
        all_confidences = sorted(all_confidences, key=lambda x: x['confidence'], reverse=True)

        # Result Details
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100

        return jsonify({
            "status": "success",
            "variety_name": "Not a Coconut" if label == "NotCoconut" else label,
            "confidence": round(confidence, 2),
            "confidence_breakdown": all_confidences, # Send this to Laravel as confidence_json
            "address": address
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
