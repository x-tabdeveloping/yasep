import catalogue
from confection import registry

registry.models = catalogue.create("confection", "models", entry_points=False)

from yasep.models.cbow import CBOW
from yasep.models.glove import GloVe
from yasep.models.skipgram import SkipGram

__all__ = ["CBOW", "GloVe", "SkipGram"]
