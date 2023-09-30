from abc import ABC
from typing import Iterable, Optional

import catalogue
from confection import Config, registry
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import (BpeTrainer, UnigramTrainer, WordLevelTrainer,
                                 WordPieceTrainer)

from yasep.doc import Document, Token
from yasep.utils import reusable


@reusable
def encode_iterable(
    tokenizer: HFTokenizer, X: Iterable[str]
) -> Iterable[list[str]]:
    if isinstance(X, str):
        raise TypeError(
            "str passed instead of iterable, did you mean to pass [X]?"
        )
    for text in X:
        encoding = tokenizer.encode(text)
        yield encoding.tokens


def tok_to_bytes(tokenizer: HFTokenizer) -> bytes:
    return tokenizer.to_str().encode("utf-8")


def bytes_to_tok(data: bytes) -> HFTokenizer:
    return HFTokenizer.from_str(data.decode("utf-8"))


class Tokenizer(ABC):
    def __call__(self, text: str) -> Document:
        encoding = self.model.encode(text)
        doc = Document(text, [])
        for index, (id, offset) in enumerate(
            zip(encoding.ids, encoding.offsets)
        ):
            token = Token(index, str(id), offset, doc)
            doc.tokens.append(token)
        return doc

    def train_from_iterable(self, X: Iterable[str], y=None):
        self.model.train_from_iterator(X, self.trainer)
        return self

    @property
    def config(self) -> Config:
        return Config(
            {
                "tokenizer": {
                    "@tokenizers": self.name,
                    **self.params,
                }
            }
        )

    @classmethod
    def from_config(cls, config: Config) -> "Tokenizer":
        resolved = registry.resolve(config)
        return resolved["tokenizer"]

    def to_bytes(self) -> bytes:
        return tok_to_bytes(self.model)

    def from_bytes(self, data: bytes):
        self.model = bytes_to_tok(data)
        return self

    @property
    def vocab(self):
        return self.model.get_vocab()

    @property
    def fitted(self):
        return bool(self.model.get_vocab_size())

    def pipe(self, texts: Iterable[str]) -> Iterable[Document]:
        @reusable
        def _pipe(texts):
            for text in texts:
                yield self.__call__(text)

        return _pipe(texts)


registry.tokenizers = catalogue.create(
    "confection", "tokenizers", entry_points=False
)


@registry.tokenizers.register("wordpiece_tokenizer.v1")
def make_wordpiece_tokenizer(
    vocab_size: int = 30000,
    min_frequency: int = 0,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=None,
    lowercase=True,
):
    return WordPieceTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )


class WordPieceTokenizer(Tokenizer):
    name = "wordpiece_tokenizer.v1"

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=True,
    ):
        self.model = HFTokenizer(WordPiece(unk_token="[UNK]"))
        self.model.pre_tokenizer = Whitespace()
        self.model.normalizer = BertNormalizer(
            clean_text, handle_chinese_chars, strip_accents, lowercase
        )
        self.params = dict(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        self.trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[UNK]"],
        )


@registry.tokenizers.register("wordlevel_tokenizer.v1")
def make_wordlevel_tokenizer(
    vocab_size: int = 30000,
    min_frequency: int = 0,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=None,
    lowercase=True,
):
    return WordLevelTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )


class WordLevelTokenizer(Tokenizer):
    name = "wordlevel_tokenizer.v1"

    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 0,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=True,
    ):
        self.model = HFTokenizer(WordLevel(unk_token="[UNK]"))
        self.model.pre_tokenizer = Whitespace()
        self.model.normalizer = BertNormalizer(
            clean_text, handle_chinese_chars, strip_accents, lowercase
        )
        self.params = dict(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        self.trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[UNK]"],
        )


@registry.tokenizers.register("unigram_tokenizer.v1")
def make_unigram_tokenizer(
    vocab_size: int = 8000,
    max_piece_length: int = 16,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=None,
    lowercase=True,
):
    return UnigramTokenizer(
        vocab_size=vocab_size,
        max_piece_length=max_piece_length,
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )


class UnigramTokenizer(Tokenizer):
    name = "unigram_tokenizer.v1"

    def __init__(
        self,
        vocab_size: int = 8000,
        max_piece_length: int = 16,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=True,
    ):
        self.model = HFTokenizer(Unigram())
        self.model.pre_tokenizer = ByteLevel()
        self.model.normalizer = BertNormalizer(
            clean_text, handle_chinese_chars, strip_accents, lowercase
        )
        self.params = dict(
            vocab_size=vocab_size,
            min_frequency=max_piece_length,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        self.trainer = UnigramTrainer(
            vocab_size=vocab_size, max_piece_length=max_piece_length
        )


@registry.tokenizers.register("bpe_tokenizer.v1")
def make_bpe_tokenizer(
    vocab_size: Optional[int] = None,
    min_frequency: Optional[int] = None,
    max_token_length: Optional[int] = None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=None,
    lowercase=True,
):
    return BpeTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        max_token_length=max_token_length,
        clean_text=clean_text,
        handle_chinese_chars=handle_chinese_chars,
        strip_accents=strip_accents,
        lowercase=lowercase,
    )


class BpeTokenizer(Tokenizer):
    name = "bpe_tokenizer.v1"

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        min_frequency: Optional[int] = None,
        max_token_length: Optional[int] = None,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=None,
        lowercase=True,
    ):
        self.model = HFTokenizer(BPE(unk_token="[UNK]"))
        self.model.pre_tokenizer = ByteLevel()
        self.model.normalizer = BertNormalizer(
            clean_text, handle_chinese_chars, strip_accents, lowercase
        )
        self.params = dict(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            max_token_length=max_token_length,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            max_token_length=max_token_length,
            min_frequency=min_frequency,
            special_tokens=["[UNK]"],
        )
