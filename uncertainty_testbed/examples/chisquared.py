from functools import partial
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
from uncertainty_testbed.generators.data_generator_explicit import AnalyticBinaryClassGenerator
from uncertainty_testbed.utilities.functions import map_to_constant

if __name__ == "__main__":
    # generate some data
    s = partial(map_to_constant, c=0.)
    eg = AnalyticBinaryClassGenerator(seed=42, num_inputs=2, name="chisq", threshold=0.5,
                                      noise_distribution="chisq", noise_scale=s)
    x, y = eg.generate_labeled(120000)
    x = x.astype("float32")
    y = y.astype("uint8")
    x_train, y_train = x[:100000, :], y[:100000]
    x_val, y_val = x[100000:, :], y[100000:]

    # train a Keras MLP
    model = keras.Sequential()
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(20, activation="relu"))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate=5e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=512,
        validation_data=(x_val, y_val)
    )

    # inspect metrics
    history_dict = history.history
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    auc = history_dict["auc"]
    val_auc = history_dict["val_auc"]
    epochs = range(1, len(loss)+1)

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "r--", label="Validation loss")
    plt.xlabel("Train Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(epochs, auc, "bo", label="Training auc")
    plt.plot(epochs, val_auc, "r--", label="Validation auc")
    plt.xlabel("Train Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.show()

    # Plot decision regions
    x_min, x_max = x_val[:, 0].min(), x_val[:, 0].max()
    y_min, y_max = x_val[:, 1].min(), x_val[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.contourf(xx, yy, z, alpha=0.4)
    ax.scatter(x_val[:1000, 0], x_val[:1000, 1], c=y_val[:1000], s=20, alpha=0.4, edgecolor="k")
    plt.show()

    model.summary()
