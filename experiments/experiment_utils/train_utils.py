"""Utilities used to train models in the experiments."""

__author__ = "nikos.daniilidis"


import copy
from tensorflow import keras
from typing import List, Tuple
import numpy as np
from experiments.model_templates.sequential_dropout_dnn_classifier import ModelScanTemplate


def generate_full(eg, n_all: int, n_train: int, n_val: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Given an event generator and sizes for all data, train data, validation data, return train and validation
    data sets.
    :param eg: Full Event Generator (features and labels)
    :param n_all: Number of observations in full data set
    :param n_train: Number of rows to use in training (count from first row)
    :param n_val: Number of rows from which to use as validation (count from first row)
    :return: Tuple of x_train, y_train, x_val, y_val
    """
    x, y = eg.generate_labeled(n_all)
    x = x.astype("float32")
    y = y.astype("uint8")
    x_train, y_train = x[:n_train, :], y[:n_train]
    assert n_all >= n_val
    assert n_train <= n_val
    x_val, y_val = x[n_val:, :], y[n_val:]
    return x_train, y_train, x_val, y_val


def opt_name(optimizer) -> str:
    """Return a standard lowercase string with the name of a keras optimizer."""
    return str(optimizer).split("object")[0].split(".")[-1].strip().lower()

optimizer_lookup = {  # dictionary mapping optimizer name to optimizer instance
    opt_name(opt): opt for opt in
    [keras.optimizers.Nadam(learning_rate=3e-5),
     keras.optimizers.SGD(learning_rate=5e-3),
     keras.optimizers.RMSprop(learning_rate=5e-3),
     keras.optimizers.Adam(learning_rate=3e-3),
     keras.optimizers.Adadelta(learning_rate=1e-1),
     keras.optimizers.Adamax(learning_rate=7e-3)]
}


def hyperparameter_scan(template: ModelScanTemplate,
                        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[dict, List[dict]]:
    """
    Perform hyperparameter scan for the model architecture and hyperparameters defined in template. Use the train and
    validation data defined in data.
    :param template:
    :param data:
    :return: Tuple[dict, List[dict]]
    """
    scan_results = []
    champion_params = None
    champion_loss = np.inf
    scan_schedule = template.scan_schedule
    x_train, y_train, x_val, y_val = data
    for params in scan_schedule:
        lr = params["learning_rate"]
        for learning_rate in [0.8 * lr, lr, 1.2 * lr]:
            optimizer = params["optimizer"]
            optimizer.learning_rate.assign(learning_rate)
            num_epochs = int(1.4 * params["epoch"] + 2)
            batch_size = int(params["batch"])
            model = template.model()
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy", "AUC"]
            )
            history = model.fit(
                x_train,
                y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
            )

            history_dict = history.history
            val_loss = history_dict["val_loss"]
            val_auc = history_dict["val_auc"]
            val_accuracy = history_dict["val_accuracy"]
            epochs = range(1, len(val_loss) + 1)

            best_loss = {"loss": val_loss[np.argmin(val_loss)], "epoch": epochs[np.argmin(val_loss)]}
            best_auc = {"auc": val_auc[np.argmax(val_auc)], "epoch": epochs[np.argmax(val_auc)]}
            best_accuracy = {"accuracy": val_accuracy[np.argmax(val_accuracy)],
                             "epoch": epochs[np.argmax(val_accuracy)]}

            ps = {k: copy.deepcopy(v) for k, v in params.items() if k not in ["optimizer", "model"]}
            ps["optimizer"] = opt_name(optimizer)
            ps["learning_rate"] = copy.deepcopy(learning_rate)
            ps["best_loss"] = best_loss
            ps["best_auc"] = best_auc
            ps["best_accuracy"] = best_accuracy
            print(ps)
            scan_results.append(ps)

            if best_loss["loss"] < champion_loss:
                champion_loss = best_loss["loss"]
                champion_params = {k: copy.deepcopy(v) for k, v in ps.items()}
                champion_params["model"] = keras.models.clone_model(model)

    return champion_params, scan_results


def train_from_params(params: dict, template: ModelScanTemplate,
                      data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) \
        -> Tuple[keras.Model, keras.callbacks.History]:
    # TODO: confirm return types
    """
    Train a model with structure defined in template, using parameters defined in params, using data.
    :param params:
    :param template:
    :param data:
    :return:
    """
    optimizer = optimizer_lookup[params["optimizer"]]
    learning_rate = params["learning_rate"]
    optimizer.learning_rate.assign(learning_rate)
    batch_size = params["batch"]
    num_epochs = params["best_loss"]["epoch"]
    x_train, y_train, x_val, y_val = data

    # train a Keras MLP
    model = template.model()
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"]
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val)
    )
    return model, history
