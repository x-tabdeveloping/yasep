from typing import Iterable, Literal

from confection import registry
from gensim.models import Word2Vec

from yasep.doc import Document
from yasep.models.model import Model
from yasep.utils import reusable


@reusable
def docs_to_ids(docs: Iterable[Document]) -> Iterable[list[str]]:
    for doc in docs:
        yield [token.id for token in doc.tokens]


@registry.models.register("cbow.v1")
def make_cbow(
    vector_size: int = 100,
    window: int = 5,
    epochs: int = 5,
    random_state: int = 0,
    negative: int = 5,
    ns_exponent: float = 0.75,
    cbow_agg: str = "mean",
    sample: float = 0.001,
    loss: str = "ns",
    batch_words: int = 10000,
    shrink_windows: bool = True,
    learning_rate: float = 0.025,
    min_learning_rate: float = 0.0001,
    n_jobs: int = 1,
):
    return CBOW(
        vector_size=vector_size,
        window=window,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        cbow_agg=cbow_agg,
        sample=sample,
        loss=loss,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


class CBOW(Model):
    name = "cbow.v1"

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        epochs: int = 5,
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_agg: Literal["mean", "sum"] = "mean",
        loss: Literal["ns", "hs"] = "ns",
        sample: float = 0.001,
        batch_words: int = 10000,
        shrink_windows: bool = True,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.params = dict(
            vector_size=vector_size,
            window=window,
            epochs=epochs,
            random_state=random_state,
            negative=negative,
            ns_exponent=ns_exponent,
            cbow_agg=cbow_agg,
            sample=sample,
            loss=loss,
            batch_words=batch_words,
            shrink_windows=shrink_windows,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            n_jobs=n_jobs,
        )
        self.model = Word2Vec(
            vector_size=vector_size,
            min_count=0,
            alpha=learning_rate,
            window=window,
            sample=sample,
            seed=random_state,
            workers=n_jobs,
            min_alpha=min_learning_rate,
            sg=0,
            hs=int(loss == "hs"),
            negative=negative,
            ns_exponent=ns_exponent,
            cbow_mean=int(cbow_agg == "mean"),
            epochs=epochs,
            trim_rule=None,
            batch_words=batch_words,
            compute_loss=True,
            shrink_windows=shrink_windows,
        )

    def train_from_iterable(self, docs: Iterable[Document]) -> "CBOW":
        ids = docs_to_ids(docs)
        self.model.build_vocab(ids)
        self.model.train(
            ids,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        self.wv = self.model.wv
        return self
