import os
# --- STEP 1: MEMORY OPTIMIZATION (MUST BE AT THE TOP) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU only for Render

import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
MODEL_PATH = "coconut_model_v2_ultra.h5" 
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# Remote Database Config (Naka-hardcode na dito base sa binigay mo)
db_config = {
    "host": "148.222.53.5",
    "user": "u914267632_group4",
    "password": "Wowgaling@12345",
    "database": "u914267632_coconutproject", 
    "port": 3306,
    "connect_timeout": 10 
}

# --- LOAD AI MODEL ---
# Ginawa nating global para isang beses lang i-load
model = None

def get_model():
    global model
    if model is None:
        try:
            print("⏳ Loading AI Model...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Model Load Error: {e}")
    return model

# I-load ang model agad pagka-start ng server
get_model()

# --- DATABASE LOGIC ---
def save_to_db(variety, confidence, address):
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = """INSERT INTO detections (variety_name, confidence, address, created_at, updated_at) 
                   VALUES (%s, %s, %s, NOW(), NOW())"""
        cursor.execute(query, (variety, confidence, address))
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ Saved to DB: {variety}")
    except Exception as e:
        print(f"❌ Database Error: {e}")

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "Server is Live", 
        "database": "Remote Connected",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    ai_model = get_model()
    if ai_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image sent"}), 400
    
    address = request.form.get('address', 'Unknown Location')

    try:
        # 1. Process Image (Memory efficient way)
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # 2. Run Prediction
        preds = ai_model.predict(img_final, verbose=0)[0]
        
        # 3. Format Predictions for UI
        top_predictions = []
        for i in range(len(CLASS_NAMES)):
            top_predictions.append({
                "label": "Not a Coconut" if CLASS_NAMES[i] == "NotCoconut" else CLASS_NAMES[i],
                "confidence": round(float(preds[i]) * 100, 2)
            })
        top_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100
        
        # 4. Handle "Not a Coconut"
        if label == "NotCoconut":
            return jsonify({
                "status": "success",
                "variety_name": "Not a Coconut",
                "confidence": round(confidence, 2),
                "address": address,
                "top_predictions": top_predictions,
                "definition": "The object does not match any known coconut seedlings."
            })

        # 5. Save to DB
        save_to_db(label, round(confidence, 2), address)

        return jsonify({
            "status": "success",
            "variety_name": label,
            "confidence": round(confidence, 2),
            "address": address,
            "top_predictions": top_predictions,
            "lifespan": "60-80 years",
            "definition": f"This is a healthy {label} seedling ready for planting."
        })

    except Exception as e:
        print(f"🔥 Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # --- IMPORTANT FOR RENDER ---
    port = int(os.environ.get("PORT", 8001)) 
    app.run(host='0.0.0.0', port=port, debug=False)
