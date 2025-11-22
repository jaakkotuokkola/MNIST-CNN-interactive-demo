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
    
    # Preprocess the image, crop to content, resize while preserving aspect,
    # place into 28x28 canvas and center via center-of-mass shift.
    arr = np.array(img).astype("float32") / 255.0

    # If background is white (mean > 0.5), invert so digit is white on black
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    
    # Have to ensure the digit is properly cropped and centered for prediction
    thresh = 0.15
    mask = arr > thresh
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        pad = 4
        r0 = max(0, r0 - pad)
        r1 = min(arr.shape[0] - 1, r1 + pad)
        c0 = max(0, c0 - pad)
        c1 = min(arr.shape[1] - 1, c1 + pad)
        cropped = arr[r0:r1+1, c0:c1+1]
    else:
        cropped = arr

    # Choose resampling filter
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = getattr(Image, 'LANCZOS', Image.BILINEAR)

    # Resize while keeping aspect ratio so the digit fits into a 20x20 box
    cropped_img = Image.fromarray(np.uint8(cropped * 255), mode='L')
    w, h = cropped_img.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round((20.0 * h) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((20.0 * w) / h)))
    resized = cropped_img.resize((new_w, new_h), resample)

    # Paste the resized digit into a 28x28 canvas centered
    canvas_img = Image.new('L', (28, 28), color=0)
    offset = ((28 - new_w) // 2, (28 - new_h) // 2)
    canvas_img.paste(resized, offset)

    arr28 = np.array(canvas_img).astype('float32') / 255.0

    # Center the digit using center of mass shifting
    total = arr28.sum()
    if total > 0:
        coords = np.indices(arr28.shape)
        cy = (coords[0] * arr28).sum() / total
        cx = (coords[1] * arr28).sum() / total
        shift_y = int(round((arr28.shape[0] - 1) / 2.0 - cy))
        shift_x = int(round((arr28.shape[1] - 1) / 2.0 - cx))
        arr28 = np.roll(arr28, shift_y, axis=0)
        arr28 = np.roll(arr28, shift_x, axis=1)

    arr = arr28
    arr = arr[..., None]
    arr = np.expand_dims(arr, 0)

    # Make predictions
    preds = MODEL.predict(arr)
    probs = preds[0].tolist()
    guess = int(np.argmax(preds[0]))

    return jsonify({"guess": guess, "probs": probs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
