from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Load models
url_model = tf.keras.models.load_model("models/url_model")
image_model = tf.keras.models.load_model("models/image_model")

# Dummy file model (replace with actual)
from sklearn.ensemble import RandomForestClassifier
import joblib
file_model = joblib.load("models/file_model/file_model.pkl")

# URL preprocessing
def preprocess_url(url):
    url = url.lower()
    max_len = 200
    x = [ord(c) for c in url[:max_len]]
    if len(x) < max_len:
        x += [0]*(max_len-len(x))
    return np.array([x])

@app.route("/scan/url", methods=["POST"])
def scan_url():
    url = request.json.get("url")
    if not url:
        return jsonify({"error":"Missing URL"}), 400
    x = preprocess_url(url)
    score = float(url_model.predict(x))
    phishing = score > 0.5
    return jsonify({"score":score, "phishing":phishing})

@app.route("/scan/image", methods=["POST"])
def scan_image():
    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"}), 400
    img_file = request.files["image"]
    from tensorflow.keras.preprocessing import image
    img = image.load_img(img_file, target_size=(224,224))
    x = image.img_to_array(img)/255.0
    x = np.expand_dims(x, axis=0)
    score = float(image_model.predict(x))
    phishing = score > 0.5
    return jsonify({"score":score, "phishing":phishing})

@app.route("/scan/file", methods=["POST"])
def scan_file():
    if "file" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    f = request.files["file"]
    # Extract features from file
    features = np.array([[0,0,0,0]])  # dummy
    score = file_model.predict_proba(features)[0][1]
    phishing = score > 0.5
    return jsonify({"score":float(score), "phishing":phishing})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
