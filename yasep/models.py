import tempfile
from typing import Iterable, Literal, Optional

import catalogue
import numpy as np
from confection import Config, registry
from gensim.models import KeyedVectors
from gensim.models import Word2Vec as GsWord2Vec
from glovpy import GloVe as GlovpyGloVe

from yasep.doc import Document
from yasep.exceptions import NotFittedError
from yasep.utils import reusable

registry.models = catalogue.create("confection", "models", entry_points=False)


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


@reusable
def docs_to_ids(docs: Iterable[Document]) -> Iterable[list[str]]:
    for doc in docs:
        yield [token.id for token in doc.tokens]


@registry.models.register("word2vec.v1")
def make_word2vec(
    vector_size: int = 100,
    window: int = 5,
    algorithm: str = "cbow",
    epochs: int = 5,
    random_state: int = 0,
    negative: int = 5,
    ns_exponent: float = 0.75,
    cbow_agg: str = "mean",
    sample: float = 0.001,
    hs: bool = False,
    batch_words: int = 10000,
    shrink_windows: bool = True,
    learning_rate: float = 0.025,
    min_learning_rate: float = 0.0001,
    n_jobs: int = 1,
):
    return Word2Vec(
        vector_size=vector_size,
        window=window,
        algorithm=algorithm,
        epochs=epochs,
        random_state=random_state,
        negative=negative,
        ns_exponent=ns_exponent,
        cbow_agg=cbow_agg,
        sample=sample,
        hs=hs,
        batch_words=batch_words,
        shrink_windows=shrink_windows,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        n_jobs=n_jobs,
    )


class Word2Vec(Model):
    name = "word2vec.v1"

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        algorithm: Literal["cbow", "sg"] = "cbow",
        epochs: int = 5,
        random_state: int = 0,
        negative: int = 5,
        ns_exponent: float = 0.75,
        cbow_agg: Literal["mean", "sum"] = "mean",
        sample: float = 0.001,
        hs: bool = False,
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
            algorithm=algorithm,
            epochs=epochs,
            random_state=random_state,
            negative=negative,
            ns_exponent=ns_exponent,
            cbow_agg=cbow_agg,
            sample=sample,
            hs=hs,
            batch_words=batch_words,
            shrink_windows=shrink_windows,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            n_jobs=n_jobs,
        )
        self.model = GsWord2Vec(
            vector_size=vector_size,
            min_count=0,
            alpha=learning_rate,
            window=window,
            sample=sample,
            seed=random_state,
            workers=n_jobs,
            min_alpha=min_learning_rate,
            sg=int(algorithm == "sg"),
            hs=int(hs),
            negative=negative,
            ns_exponent=ns_exponent,
            cbow_mean=int(cbow_agg == "mean"),
            epochs=epochs,
            trim_rule=None,
            batch_words=batch_words,
            compute_loss=True,
            shrink_windows=shrink_windows,
        )

    def train_from_iterable(self, docs: Iterable[Document]) -> "Word2Vec":
        ids = docs_to_ids(docs)
        self.model.build_vocab(ids)
        self.model.train(
            ids,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs,
        )
        self.wv = self.model.wv
        return self


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
