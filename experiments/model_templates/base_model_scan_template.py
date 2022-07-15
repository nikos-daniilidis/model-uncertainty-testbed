"""Base class for model scan templates."""

__author__ = "nikos.daniilidis"

from tensorflow import keras
from typing import List


class ModelScanTemplate(object):
    """Convenience class which binds together a model architecture and the parameters for a scan."""
    def model(self) -> keras.Model:
        raise NotImplementedError

    def scan_schedule(self) -> List[dict]:
        raise NotImplementedError
