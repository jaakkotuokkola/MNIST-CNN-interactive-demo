from flask import Flask, request, jsonify, send_from_directory
import os
import io
import base64
import numpy as np
from PIL import Image

app = Flask(__name__, static_folder="static")

def load_model():
    import tensorflow as tf
    model_path = os.path.join("model", "mnist_cnn.h5")
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

MODEL = load_model()

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
        if MODEL is None:
            return jsonify({"error": "Model not found. Run model/train_mnist.py to create model/mnist_cnn.h5"}), 400

    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    img_data = data["image"].split(",")[-1]
    try:
        decoded = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(decoded)).convert("L")
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400
    
    # Preprocess the image
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = getattr(Image, 'LANCZOS', Image.BILINEAR)
    img = img.resize((28, 28), resample)
    arr = np.array(img).astype("float32") / 255.0

    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = arr[..., None]
    arr = np.expand_dims(arr, 0)

    # Make predictions
    preds = MODEL.predict(arr)
    probs = preds[0].tolist()
    guess = int(np.argmax(preds[0]))

    return jsonify({"guess": guess, "probs": probs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
