The file train_mnist.py trains a convolutional neural network on the MNIST handwritten digit dataset. The app allows you to draw a digit and tasks the neural network with recognizing it. This is a classic machine learning problem, with roots dating back to the 1980s and 1990s. It has long served as a benchmark for evaluating image classification algorithms and neural networks.

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train the model:

```bash
python3 model/train_mnist.py --epochs 5
```

3. Run the app:

```bash
python3 app.py
```

4. Open `http://localhost:5000` in your browser, draw a digit, and click `Predict`.

Notes

- The server will look for `model/mnist_cnn.h5`. If you haven't trained it, run the training script first.
- The UI sends the canvas PNG as a base64 data URL, the server preprocesses it to 28x28 grayscale and predicts.
- Have to make some changes so the drawn digit is fully and accurately captured for the model before prediction.
