from typing import Callable, Iterable, Optional, Union

import numpy as np
import safetensors.numpy
from confection import Config, registry

from yasep.doc import Document
from yasep.exceptions import NotFittedError
from yasep.utils import reusable


@registry.models.register("static_model.v1")
def make_static():
    return Model()


ArraySequence = Union[np.ndarray, list[np.ndarray]]


class Model:
    name = "static_model.v1"

    def __init__(self, embeddings: Optional[np.ndarray] = None):
        self.embeddings = embeddings
        self.params = dict()

    def __call__(self, doc: Document):
        if self.embeddings is None:
            raise NotFittedError("Model has not been fitted yet.")
        vectors = np.take(self.embeddings, doc.ids, axis=0)
        doc.vectors = vectors
        return doc

    def encode(
        self,
        ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        if self.embeddings is None:
            raise NotFittedError("Model has not been fitted yet.")
        vectors = np.take(self.embeddings, ids, axis=0)
        vectors = (vectors.T * attention_mask).T
        vectors_sum = np.sum(vectors, axis=0)
        n_nonmask = np.sum(attention_mask)
        return vectors_sum / n_nonmask

    def encode_batch(
        self,
        ids: ArraySequence,
        attention_mask: ArraySequence,
    ):
        if self.embeddings is None:
            raise NotFittedError("Model has not been fitted yet.")
        if isinstance(ids, list):
            res = []
            for id, att in zip(ids, attention_mask):
                res.append(self.encode(id, att))
            return np.stack(res)
        else:
            vectors = np.take(self.embeddings, ids, axis=0)
            masked = np.transpose(
                (np.transpose(vectors, (2, 0, 1)) * attention_mask), (1, 2, 0)
            )
            summed = np.sum(masked, axis=1)
            n_nonmask = np.sum(masked, axis=1)
            return summed / n_nonmask

    def pipe(self, docs: Iterable[Document]) -> Iterable[Document]:
        @reusable
        def _pipe(docs):
            for doc in docs:
                yield self.__call__(doc)

        return _pipe(docs)

    @property
    def config(self) -> Config:
        return Config(
            {
                "model": {
                    "@models": self.name,
                    **self.params,
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "Model":
        resolved = registry.resolve(config)
        return resolved["tokenizer"]

    def to_bytes(self) -> bytes:
        if self.embeddings is None:
            raise NotFittedError(
                "Can't save model if it hasn't been fitted yet."
            )
        return safetensors.numpy.save({"embeddings": self.embeddings})

    def from_bytes(self, data: bytes):
        tensor_dict = safetensors.numpy.load(data)
        self.embeddings = tensor_dict["embeddings"]
        return self

    @property
    def fitted(self):
        return self.embeddings is not None

    def train_from_iterable(self, texts: Iterable[Document]) -> "Model":
        return self
