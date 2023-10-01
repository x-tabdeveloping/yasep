import tempfile
from typing import Iterable, Optional

import numpy as np
import safetensors.numpy
from confection import Config, registry
from gensim.models import KeyedVectors

from yasep.doc import Document
from yasep.exceptions import NotFittedError
from yasep.utils import reusable


@registry.models.register("static_model.v1")
def make_static():
    return Model()


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
