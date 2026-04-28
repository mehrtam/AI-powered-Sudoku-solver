"""
Train the digit-recognition CNN used by the Sudoku Solver.

Trains on MNIST with augmentation (rotation, zoom, shifts), early stopping on
validation loss, and saves the final model to disk.

Usage:
    python train_model.py --output sudoku_model.keras
"""

import argparse

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 28, 28, 1)
    x_test = (x_test / 255.0).reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--output", default="sudoku_model.keras",
                        help="Where to save the trained model.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3,
                        help="EarlyStopping patience on val_loss.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    x_train, y_train, x_test, y_test = load_data()
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=args.seed
    )

    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    datagen.fit(x_tr)

    model = build_model()
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True
    )
    model.fit(
        datagen.flow(x_tr, y_tr, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping],
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_acc:.4f}  (loss: {test_loss:.4f})")

    model.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
