import tempfile
from typing import Iterable, Optional

import numpy as np
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

    def __init__(self, wv: Optional[KeyedVectors] = None):
        self.wv = wv
        self.params = dict()

    def __call__(self, doc: Document):
        if self.wv is None:
            raise NotFittedError("Model has not been fitted yet.")
        n_toks = len(doc.tokens)
        vec_size = self.wv.vector_size
        vectors = np.empty((n_toks, vec_size))
        for token in doc.tokens:
            try:
                vectors[token.index] = self.wv[token.id]
            except KeyError:
                vectors[token.index] = np.full(vec_size, np.nan)
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
        if self.wv is None:
            raise NotFittedError(
                "Can't save model if it hasn't been fitted yet."
            )
        with tempfile.NamedTemporaryFile(prefix="model_") as tmp:
            temporary_filepath = tmp.name
            self.wv.save(temporary_filepath)
            with open(temporary_filepath, "rb") as temp_buffer:
                return temp_buffer.read()

    def from_bytes(self, data: bytes):
        with tempfile.NamedTemporaryFile(prefix="model_") as tmp:
            tmp.write(data)
            self.wv = KeyedVectors.load(tmp.name)
        return self

    @property
    def fitted(self):
        return self.wv is not None

    def train_from_iterable(self, texts: Iterable[Document]) -> "Model":
        return self
