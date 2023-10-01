from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


@dataclass
class Token:
    index: int
    offset: tuple[int, int]
    doc: "Document"

    @property
    def orth(self) -> str:
        start, end = self.offset
        return self.doc.text[start:end]

    @property
    def vector(self) -> np.ndarray:
        if self.doc.vectors is None:
            raise TypeError("No vectors have been assigned yet.")
        return self.doc.vectors[self.index]

    @property
    def id(self) -> int:
        return self.doc.ids[self.index]

    def __repr__(self) -> str:
        return f"Token({self.id}: '{self.orth}')"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class Document:
    text: str
    tokens: list[Token]
    ids: np.ndarray
    vectors: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        text_start = self.text[:50]
        return f"Document('{text_start}...')"

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self) -> Iterable[Token]:
        return iter(self.tokens)
