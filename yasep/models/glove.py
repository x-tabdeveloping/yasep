from typing import Iterable

import numpy as np
from confection import registry
from glovpy import GloVe as GlovpyGloVe

from yasep.doc import Document
from yasep.models.model import Model
from yasep.utils import reusable


@reusable
def docs_to_ids(docs: Iterable[Document]) -> Iterable[list[str]]:
    for doc in docs:
        yield [str(token.id) for token in doc.tokens]


def kv_to_array(kv) -> np.ndarray:
    vocab = []
    for id in kv.index_to_key:
        try:
            vocab.append(int(id))
        except ValueError:
            continue
    max_token = max(vocab)
    unk_vector = kv.get_mean_vector(kv.index_to_key[:100])
    vector_size = kv.vector_size
    embeddings = np.copy(
        np.broadcast_to(unk_vector, (max_token + 1, vector_size))
    )
    for id in vocab:
        embeddings[id] = kv[str(id)]
    return embeddings


@registry.models.register("glove.v1")
def make_glove(
    vector_size: int = 100,
    alpha: float = 0.75,
    window: int = 15,
    symmetric: bool = True,
    distance_weighting: bool = True,
    iter: int = 25,
    initial_learning_rate: float = 0.05,
    n_jobs: int = 8,
    memory: float = 4.0,
):
    return GloVe(
        vector_size=vector_size,
        alpha=alpha,
        window=window,
        symmetric=symmetric,
        distance_weighting=distance_weighting,
        iter=iter,
        initial_learning_rate=initial_learning_rate,
        n_jobs=n_jobs,
        memory=memory,
    )


class GloVe(Model):
    name = "glove.v1"

    def __init__(
        self,
        vector_size: int = 100,
        alpha: float = 0.75,
        window: int = 15,
        symmetric: bool = True,
        distance_weighting: bool = True,
        iter: int = 25,
        initial_learning_rate: float = 0.05,
        n_jobs: int = 8,
        memory: float = 4.0,
    ):
        super().__init__()
        self.params = dict(
            vector_size=vector_size,
            alpha=alpha,
            window=window,
            symmetric=symmetric,
            distance_weighting=distance_weighting,
            iter=iter,
            initial_learning_rate=initial_learning_rate,
            n_jobs=n_jobs,
            memory=memory,
        )
        self.model = GlovpyGloVe(
            vector_size=vector_size,
            alpha=alpha,
            window_size=window,
            symmetric=symmetric,
            distance_weighting=distance_weighting,
            iter=iter,
            initial_learning_rate=initial_learning_rate,
            threads=n_jobs,
            memory=memory,
        )

    def train_from_iterable(self, docs: Iterable[Document]) -> "GloVe":
        ids = docs_to_ids(docs)
        self.model.train(ids)
        self.embeddings = kv_to_array(self.model.wv)
        return self
