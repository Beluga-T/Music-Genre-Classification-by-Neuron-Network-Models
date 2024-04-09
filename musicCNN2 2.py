import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from MUSIC.music import predict

DATA_PATH = "data.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x, y

def prepare_datasets(test_size, validation_size):

    # load data
    x, y = load_data(DATA_PATH)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=validation_size)

    # 3D array
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # First Layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=input_shape))

    # Second Layer
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))

    # Max Pooling Layer
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    # Third Layer
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

    # Fourth Layer
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))

    # Max Pooling Layer
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    # fifth Layer
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))

    # Sixth Layer
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))

    # Max Pooling Layer
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

    # Flattening Layer
    model.add(keras.layers.Flatten())

    # Dropout Layer
    model.add(keras.layers.Dropout(0.5, noise_shape=None, seed=None))

    # Adding the first fully connected layer
    model.add(keras.layers.Dense(units=128, activation='relu'))

    # Output Layer
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    return model



    x = x[np.newaxis, ...]
    prediction = model.predict(x)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))

def plot_history(history):

    fig, axs = plt.subplots(2)

    #create accuracy subplot
    axs[0].plot(history.history["accuracy"], label = "train accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc = "lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="lower right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":

    # create train, validation and test set
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # build the CNN net
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_model(input_shape)
    keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=96,


    )
    # Train the CNN
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()


    keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


    # Train the CNN
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=36, epochs=30)

    # Evaluate the CNN on test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is : {}".format(test_accuracy))

    # make prediction
    x = x_test[100]
    y = y_test[100]
    predict(model, x, y)

    plot_history(history)