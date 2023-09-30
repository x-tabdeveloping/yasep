import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Union

import numpy as np
from confection import Config, registry
from huggingface_hub import HfApi, snapshot_download

from yasep.doc import Document
from yasep.exceptions import NotFittedError
from yasep.hub import DEFAULT_README
from yasep.models import Model
from yasep.tokenizers import Tokenizer
from yasep.utils import reusable


@dataclass
class Pipeline:
    tokenizer: Tokenizer
    model: Model

    @property
    def fitted(self) -> bool:
        return self.tokenizer.fitted and self.model.fitted

    def __call__(self, text: str) -> Document:
        if not self.fitted:
            raise NotFittedError("Pipeline has not been fitted yet!")

        doc = self.tokenizer(text)
        doc = self.model(doc)
        return doc

    def pipe(self, texts: Iterable[str]) -> Iterable[Document]:
        docs = self.tokenizer.pipe(texts)
        return self.model.pipe(docs)

    def train_from_iterator(self, corpus: Iterable[str]) -> "Pipeline":
        self.tokenizer.train_from_iterable(corpus)
        docs = self.tokenizer.pipe(corpus)
        self.model.train_from_iterable(docs)
        return self

    def encode(self, text: str, agg: Callable = np.nanmean) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError(
                "text is not type str. Did you mean to call encode_batch?"
            )
        doc = self.__call__(text)
        return agg(doc.vectors, axis=0)

    def encode_batch(
        self, texts: Iterable[str], agg: Callable = np.nanmean
    ) -> np.ndarray:
        embeddings = []
        for text in texts:
            embeddings.append(self.encode(text, agg))
        return np.stack(embeddings)

    @property
    def config(self) -> Config:
        return self.tokenizer.config.merge(self.model.config)

    @classmethod
    def from_config(cls, config: Config) -> "Pipeline":
        resolved = registry.resolve(config)
        tokenizer = resolved["tokenizer"]
        model = resolved["model"]
        return cls(tokenizer=tokenizer, model=model)

    def to_disk(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)
        config_path = path.joinpath("config.cfg")
        tokenizer_path = path.joinpath("tokenizer.bin")
        model_path = path.joinpath("model.bin")
        with open(model_path, "wb") as model_file:
            model_file.write(self.model.to_bytes())
        with open(tokenizer_path, "wb") as tokenizer_file:
            tokenizer_file.write(self.tokenizer.to_bytes())
        self.config.to_disk(config_path)

    @classmethod
    def from_disk(cls, path: Union[str, Path]) -> "Pipeline":
        path = Path(path)
        config_path = path.joinpath("config.cfg")
        tokenizer_path = path.joinpath("tokenizer.bin")
        model_path = path.joinpath("model.bin")
        config = Config().from_disk(config_path)
        resolved = registry.resolve(config)
        with open(tokenizer_path, "rb") as tokenizer_file:
            tokenizer = resolved["tokenizer"].from_bytes(tokenizer_file.read())
        with open(model_path, "rb") as model_file:
            model = resolved["model"].from_bytes(model_file.read())
        return cls(tokenizer=tokenizer, model=model)

    def to_hub(self, repo_id: str, add_readme: bool = True) -> None:
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.to_disk(tmp_dir)
            if add_readme:
                with open(
                    Path(tmp_dir).joinpath("README.md"), "w"
                ) as readme_f:
                    readme_f.write(DEFAULT_README.format(repo=repo_id))
            api.upload_folder(
                folder_path=tmp_dir, repo_id=repo_id, repo_type="model"
            )

    @classmethod
    def from_hub(cls, repo_id: str) -> "Pipeline":
        in_dir = snapshot_download(repo_id=repo_id)
        res = cls.from_disk(in_dir)
        return res
