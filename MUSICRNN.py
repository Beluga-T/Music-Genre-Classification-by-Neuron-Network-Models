import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

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

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):

    # create model
    model = keras.Sequential()

    # LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, x, y):

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
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape)

    # Train the CNN
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    # Train the CNN
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=36, epochs=50)

    # Evaluate the CNN on test set
    test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy on test set is : {}".format(test_accuracy))

    # make prediction
    x = x_test[100]
    y = y_test[100]
    predict(model, x, y)

    plot_history(history)