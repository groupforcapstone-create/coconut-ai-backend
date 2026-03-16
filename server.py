import os
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CLOUD CONFIGURATION ---
# The .h5 file must be in the same root folder on GitHub
MODEL_PATH = "coconut_model_v2_ultra.h5" 
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# Remote Database Config (Based on your .env file)
db_config = {
    "host": "148.222.53.5",          # Your Hostinger Public IP
    "user": "u914267632_group4",      # Your Database Username
    "password": "Wowgaling@12345",    # Your Database Password
    "database": "u914267632_coconutproject", 
    "port": 3306,
    "connect_timeout": 10             # Prevents long hangs if firewall blocks connection
}

# --- LOAD AI MODEL (Cloud Efficient) ---
def load_model_file():
    try:
        # Using compile=False saves memory on Render's 512MB RAM limit
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Model Load Error: {e}")
        return None

model = load_model_file()

# --- DATABASE LOGIC ---
def save_to_db(variety, confidence, address):
    """Saves scan results directly to your Hostinger MySQL database"""
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Ensure your table 'detections' has these exact column names
        query = """INSERT INTO detections (variety_name, confidence, address, created_at, updated_at) 
                   VALUES (%s, %s, %s, NOW(), NOW())"""
        
        cursor.execute(query, (variety, confidence, address))
        connection.commit()
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"Database Error: {e}")

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    """Verify that the server is online"""
    return jsonify({"status": "Server is Live", "database": "Remote Connected"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on server"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No image sent"}), 400
    
    # Receive address string from Flutter GPS logic
    address = request.form.get('address', 'Unknown Location')

    try:
        # Convert uploaded file to OpenCV format
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Pre-process image for AI
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # Run Prediction
        preds = model.predict(img_final, verbose=0)[0]
        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100

        # Handle Non-Coconut items
        if label == "NotCoconut":
            return jsonify({
                "variety_name": "Not a Coconut",
                "confidence": confidence,
                "address": address
            })

        # Save to your Online Database (Hostinger)
        save_to_db(label, confidence, address)

        return jsonify({
            "variety_name": label,
            "confidence": round(confidence, 2),
            "address": address
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- RENDER DEPLOYMENT SETTING ---
if __name__ == "__main__":
    # Render assigns a dynamic port, we must use os.environ to get it
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)