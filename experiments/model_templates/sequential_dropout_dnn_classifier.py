"""Class definitions keeping all the parameters for model architectures and hyperparameter scans in one place."""

__author__ = "nikos.daniilidis"

from abc import ABC
from tensorflow import keras
from typing import List
from experiments.model_templates.base_model_scan_template import ModelScanTemplate
from uncertainty_testbed.uncertainty_models.mc_dropout import MCDropoutLayer, MCDropoutKerasClassification


def build_mcdropout_model(optimizer) -> keras.Model:
    """Function which builds the model corresponding to the scan template in the corresponding ScanTemplate class.
    Note that this model uses MCDropout layer a d not Dropout layer. Currently it is up to the user to ensure the
    model specification is the same in both places."""
    layers = (
        keras.layers.Dense(60, activation="relu"),
        MCDropoutLayer(0.25),
        keras.layers.Dense(60, activation="relu"),
        MCDropoutLayer(0.25),
        keras.layers.Dense(60, activation="relu"),
        MCDropoutLayer(0.25),
        keras.layers.Dense(1, activation="sigmoid")
    )
    model = MCDropoutKerasClassification(
        layer_tuple=layers,
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=("accuracy", "AUC"),
        name="MCDropout"
    )
    return model


class SequentialDropout_10_60_025_x3_ScanTemplate(ModelScanTemplate, ABC):
    """Convenience class which binds together a model architecture and the parameters for a scan."""
    def __init__(self):
        self.scan_schedule = self.scan_schedule()

    def model(self):
        """Definition of the model"""
        model = keras.Sequential()
        model.add(keras.layers.Dense(60, activation="relu"))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(60, activation="relu"))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(60, activation="relu"))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        return model

    def scan_schedule(self) -> List[dict]:
        scan_schedule = [
            # preset values to guide the hyperparameter scan for 10 dimensional gaussian feature space.
            {
                "optimizer": keras.optimizers.Nadam(learning_rate=3e-5),
                "learning_rate": 3e-5,
                "batch": 256,
                "epoch": 3 * 28
            },
            {
                "optimizer": keras.optimizers.SGD(learning_rate=5e-3),
                "learning_rate": 5e-3,
                "batch": 256,
                "epoch": 3 * 59
            },
            {
                "optimizer": keras.optimizers.SGD(learning_rate=1e-2),
                "learning_rate": 1e-2,
                "batch": 8,
                "epoch": 3 * 1
            },
            {
                "optimizer": keras.optimizers.RMSprop(learning_rate=5e-3),
                "learning_rate": 5e-3,
                "batch": 8,
                "epoch": 3 * 1  # 0.1781
            },
            {
                "optimizer": keras.optimizers.Adam(learning_rate=3e-3),
                "learning_rate": 3e-3,
                "batch": 8,
                "epoch": 3 * 1  # 0.2076
            },
            {
                "optimizer": keras.optimizers.Adadelta(learning_rate=1e-1),
                "learning_rate": 1e-1,
                "batch": 8,
                "epoch": 3 * 2  # 0.2354
            },
            {
                "optimizer": keras.optimizers.Adamax(learning_rate=7e-3),
                "learning_rate": 7e-3,
                "batch": 8,
                "epoch": 3 * 1  # 0.2249
            },
            {
                "optimizer": keras.optimizers.Nadam(learning_rate=7e-3),
                "learning_rate": 7e-3,
                "batch": 8,
                "epoch": 3 * 4  # 0.1905
            }
        ]
        return scan_schedule