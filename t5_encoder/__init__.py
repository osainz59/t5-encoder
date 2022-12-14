from transformers.models.auto.modeling_auto import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    _LazyAutoMapping
)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from .modeling_t5 import T5ForTokenClassification, T5ForSequenceClassification

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES["t5"] = "T5ForTokenClassification"
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES["t5"] = "T5ForSequenceClassification"

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)

from transformers.models import t5

setattr(t5, "T5ForTokenClassification", T5ForTokenClassification)
setattr(t5, "T5ForSequenceClassification", T5ForSequenceClassification)

__version__ = "0.1"

__all__ = [
    "T5ForTokenClassification",
    "T5ForSequenceClassification"
]