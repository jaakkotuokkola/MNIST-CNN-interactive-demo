import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Trains a simple convolutional neural network on the MNIST dataset

def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main(epochs=5):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = build_model(input_shape=(28, 28, 1))
    model.summary()
    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1)

    os.makedirs("model", exist_ok=True)
    save_path = os.path.join("model", "mnist_cnn.h5")
    model.save(save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(epochs=args.epochs)