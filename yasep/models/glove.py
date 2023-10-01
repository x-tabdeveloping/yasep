from typing import Iterable

from confection import registry
from glovpy import GloVe as GlovpyGloVe

from yasep.doc import Document
from yasep.models.model import Model
from yasep.utils import reusable


@reusable
def docs_to_ids(docs: Iterable[Document]) -> Iterable[list[str]]:
    for doc in docs:
        yield [token.id for token in doc.tokens]


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
        self.wv = self.model.wv
        return self
