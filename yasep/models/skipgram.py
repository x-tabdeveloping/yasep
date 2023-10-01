from typing import Iterable, Literal

import numpy as np
from confection import registry
from gensim.models import Word2Vec

from yasep.doc import Document
from yasep.models.model import Model
from yasep.utils import reusable


@reusable
def docs_to_ids(docs: Iterable[Document]) -> Iterable[list[str]]:
    for doc in docs:
        yield [str(token.id) for token in doc.tokens]


def kv_to_array(kv) -> np.ndarray:
    vocab = list(map(int, kv.index_to_key))
    max_token = max(vocab)
    unk_vector = kv.get_mean_vector(kv.index_to_key[:100])
    vector_size = kv.vector_size
    embeddings = np.copy(
        np.broadcast_to(unk_vector, (max_token + 1, vector_size))
    )
    for key in kv.index_to_key:
        embeddings[int(key)] = kv[key]
    return embeddings


@registry.models.register("skip_gram.v1")
def make_skip_gram(
    vector_size: int = 100,
    window: int = 5,
    epochs: int = 5,
    random_state: int = 0,
    negative: int = 5,
    ns_exponent: float = 0.75,
    sample: float = 0.001,
    loss: str = "ns",
    batch_words: int = 10000,
    shrink_windows: bool = True,
    learning_rate: float = 0.025,
    min_learning_rate: float = 0.0001,
    n_jobs: int = 1,
):
    return SkipGram(
        vector_size=vector_size,
        window=window,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        sample=sample,
        loss=loss,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


class SkipGram(Model):
    name = "skip_gram.v1"

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        epochs: int = 5,
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        skip_gram_agg: Literal["mean", "sum"] = "mean",
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
            skip_gram_agg=skip_gram_agg,
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
            epochs=epochs,
            trim_rule=None,
            batch_words=batch_words,
            compute_loss=True,
            shrink_windows=shrink_windows,
        )

    def train_from_iterable(self, docs: Iterable[Document]) -> "SkipGram":
        ids = docs_to_ids(docs)
        self.model.build_vocab(ids)
        self.model.train(
            ids,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        self.embeddings = kv_to_array(self.model.wv)
        return self
