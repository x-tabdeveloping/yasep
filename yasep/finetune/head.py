from abc import ABC, abstractmethod, abstractproperty
from typing import Sequence, Union

import jax
import numpy as np

Array = Union[jax.Array, np.ndarray]


class Head(ABC):
    @abstractmethod
    def initialize(self, encodings: Array, labels: Sequence):
        pass

    @abstractproperty
    def params(self) -> dict:
        pass

    @abstractmethod
    def loss(self, encodings: Array, outputs: Array, **params) -> float:
        pass

    @abstractmethod
    def transform_outputs(self, labels: Sequence) -> Array:
        pass

    def predict(self, encodings: Array) -> Sequence:
        pass
