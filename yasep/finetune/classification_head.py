from typing import Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.nn import relu, softmax
from optax import softmax_cross_entropy

from yasep.doc import Document
from yasep.finetune.head import Head

Array = Union[jax.Array, np.ndarray]
ArraySequence = Union[Array, list[np.ndarray]]


class ClassificationHead(Head):
    def __init__(self, hidden_dim: int = 100, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.initialised = False
        self.seed = seed

    def initialize(self, encodings: Array, outputs: Sequence):
        self.classes = np.unique(outputs)
        self.class_to_id = {label: id for id, label in enumerate(self.classes)}
        self.initialised = True
        self.out_dim = len(self.classes)
        self.input_dim = encodings.shape[1]
        rng = jax.random.PRNGKey(self.seed)
        b1_rng, b2_rng, w1_rng, w2_rng = jax.random.split(rng, num=4)
        self.w1 = jax.random.normal(w1_rng, (self.input_dim, self.hidden_dim))
        self.w2 = jax.random.normal(w2_rng, (self.hidden_dim, self.out_dim))
        self.b1 = jax.random.normal(b1_rng, (self.input_dim, self.hidden_dim))
        self.b2 = jax.random.normal(b2_rng, (self.hidden_dim, self.out_dim))

    @property
    def params(self):
        return dict(b1=self.b1, b2=self.b2, w1=self.w1, w2=self.w2)

    def update_params(
        self,
        b1: Array,
        b2: Array,
        w1: Array,
        w2: Array,
    ):
        self.b1 = b1
        self.b2 = b2
        self.w1 = w1
        self.w2 = w2

    def apply(
        self,
        encodings: Array,
        b1: Array,
        b2: Array,
        w1: Array,
        w2: Array,
    ) -> Array:
        """Returns unnormalized logits."""
        hidden = relu(b1 + encodings @ w1)
        out = relu(b2 + hidden @ w2)
        return out

    def loss(self, encodings: Array, outputs: Array, **params) -> float:
        """Returns mean softmax cross entropy loss for a batch"""
        logits = self.apply(encodings, **params)
        return jnp.mean(softmax_cross_entropy(logits, outputs))  # type: ignore

    def transform_outputs(self, labels: Sequence) -> Array:
        ids = np.array([self.class_to_id[label] for label in labels])
        return jax.nn.one_hot(ids, self.out_dim)

    def predict(self, encodings: Array) -> Sequence:
        logits = self.apply(encodings, **self.params)
        probs = softmax(logits)
        ids = jnp.argmax(probs, axis=1)
        return self.classes[ids]
