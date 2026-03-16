import os
# --- STEP 1: MEMORY OPTIMIZATION (MUST BE AT THE TOP) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = "coconut_model_v2_ultra.h5" 
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# --- DATABASE CONFIG (FROM ENV VARIABLES) ---
db_config = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME"),
    "port": int(os.environ.get("DB_PORT", 3306)),
    "connect_timeout": 10
}

# --- LOAD AI MODEL ---
def load_model_file():
    try:
        if os.path.exists(MODEL_PATH):
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            print(f"Model file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Model Load Error: {e}")
        return None

model = load_model_file()

# --- DATABASE LOGIC ---
def save_to_db(variety, confidence, address):
    """Saves scan results to MySQL database"""
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        query = """INSERT INTO detections (variety_name, confidence, address, created_at, updated_at) 
                   VALUES (%s, %s, %s, NOW(), NOW())"""
        
        cursor.execute(query, (variety, confidence, address))
        connection.commit()
        cursor.close()
        connection.close()
        print(f"Saved to DB: {variety} - {confidence}%")
    except Exception as e:
        print(f"Database Error: {e}")

# --- API ROUTES ---
@app.route("/", methods=["GET"])
def health_check():
    model_status = "Ready" if model else "Failed to Load"
    return jsonify({
        "status": "AI Server Live",
        "model": model_status,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "AI Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    address = request.form.get('address', 'Unknown Location')
    
    try:
        file = request.files['file']
        
        # Read file without saving to disk
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
        
        # Build confidence breakdown
        confidence_json = []
        for i, name in enumerate(CLASS_NAMES):
            confidence_json.append({
                "label": "Not a Coconut" if name == "NotCoconut" else name,
                "confidence": round(float(preds[i]) * 100, 2)
            })
        confidence_json = sorted(confidence_json, key=lambda x: x['confidence'], reverse=True)
        
        label = CLASS_NAMES[idx]
        confidence = round(float(preds[idx]) * 100, 2)
        
        # Save to database if it's a coconut
        if label != "NotCoconut":
            save_to_db(label, confidence, address)
        
        return jsonify({
            "status": "success",
            "variety_name": "Not a Coconut" if label == "NotCoconut" else label,
            "confidence": confidence,
            "confidence_json": confidence_json,
            "address": address
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
