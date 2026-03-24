import os
# --- STEP 1: MEMORY OPTIMIZATION (MUST BE AT THE TOP) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force CPU only for Render Free Tier

import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
# Gamitin ang .tflite file para iwas crash sa Render
MODEL_PATH = "coconut_model.tflite" 
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# Remote Database Config (Hostinger MySQL)
db_config = {
    "host": "148.222.53.5",
    "user": "u914267632_group4",
    "password": "Wowgaling@12345",
    "database": "u914267632_coconutproject", 
    "port": 3306,
    "connect_timeout": 10 
}

# --- LOAD TFLITE MODEL ---
print("⏳ Loading TFLite Model... very lightweight!")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Kunin ang input at output details ng model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite Model loaded successfully!")
except Exception as e:
    print(f"❌ TFLite Load Error: {e}")
    interpreter = None

# --- DATABASE LOGIC (ASYNC) ---
def save_to_db_worker(variety, confidence, address):
    """Saves to DB in the background to prevent API lag"""
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = """INSERT INTO detections (variety_name, confidence, address, created_at, updated_at) 
                   VALUES (%s, %s, %s, NOW(), NOW())"""
        cursor.execute(query, (variety, confidence, address))
        connection.commit()
        cursor.close()
        connection.close()
        print(f"✅ DB Sync Success: {variety} logged.")
    except Exception as e:
        print(f"❌ Database Sync Error: {e}")

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success", 
        "ai_model_status": "Live (TFLite)",
        "database": "Remote Connected",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image sent"}), 400
    
    address = request.form.get('address', 'Unknown Location')

    try:
        # 1. Process Image
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # 2. Run TFLite Prediction (Inference)
        interpreter.set_tensor(input_details[0]['index'], img_final)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        # 3. Format Predictions
        top_predictions = []
        for i in range(len(CLASS_NAMES)):
            top_predictions.append({
                "label": "Not a Coconut" if CLASS_NAMES[i] == "NotCoconut" else CLASS_NAMES[i],
                "confidence": round(float(preds[i]) * 100, 2)
            })
        top_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = round(float(preds[idx]) * 100, 2)
        
        # 4. Filter "Not a Coconut"
        if label == "NotCoconut":
            return jsonify({
                "status": "success",
                "variety_name": "Not a Coconut",
                "confidence": confidence,
                "address": address,
                "top_predictions": top_predictions,
                "definition": "The object does not match any known coconut seedlings."
            })

        # 5. Background DB Save
        threading.Thread(target=save_to_db_worker, args=(label, confidence, address)).start()

        # 6. Final Response
        return jsonify({
            "status": "success",
            "variety_name": label,
            "confidence": confidence,
            "address": address,
            "top_predictions": top_predictions,
            "lifespan": "60-80 years",
            "definition": f"This is a healthy {label} seedling ready for planting."
        })

    except Exception as e:
        print(f"🔥 Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Gamitin ang PORT environment variable para sa Render
    port = int(os.environ.get("PORT", 10000)) 
    print(f"🚀 TFLite AI Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
