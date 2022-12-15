# T5 Encoder-only extension for Transformers
This repository contains the implementation of `T5ForSequenceClassification` and `T5ForTokenClassification` fully compatible with [Transformers](https://github.com/huggingface/transformers) library. While this could be a feature from the library itself is not implemented yet, so this repository contains the code for preliminary experiments before being actually included to the library.

This implementation is inspired by [EncT5: A Framework for Fine-tuning T5 as Non-autoregressive Models](https://arxiv.org/abs/2110.08426) and [A Universal Discriminator for Zero-Shot Generalization](https://arxiv.org/pdf/2211.08099.pdf) that made use of T5 encoder only.

## Installation and use
You can simply install this library by running the following command:
```bash
python -m pip install git+https://github.com/osainz59/t5-encoder
```
To use the implemented classes you have to simply import `t5_encoders` along with transformers. Example:
```python
import t5_encoders
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
model = AutoModelForSequenceClassification.from_pretrained("google/t5-v1_1-base")

outputs = model(**tokenizer("This is a sentence to classify.", return_tensors="pt"))
print(outputs.logits)
>>> tensor([[ 0.0512, -0.0594]], grad_fn=<AddmmBackward0>)
```

## GLUE results

| Model | CoLA | SST2 | MRPC | STSb | QQP | MNLI | QNLI | RTE | WNLI |
|:------|:------|:-----|:-----|:-----|:----|:-----|:-----|:----|:-----|
| RoBERTa<sub>large</sub>  | **68.0** | **96.4** | 90.9 | 92.4 | **92.2** | 90.2/90.2 | 94.7 | 86.6 | **91.3** |
| T5<sub>large</sub> | 61.2 | 96.3 | 92.4 | 89.9 | 89.9 | 89.9/89.6 | **94.8** | 87.2 | 85.6 | 
| T5-Enc<sub>large</sub> | 55.0 | 96.1 | **93.3** | **92.7** | 91.4 | **90.5/90.4** | 94.7 | **88.8** | 47.9 |

## NER results
| Model | CoNLL-2003 (F1) |
|:------|:------|
|[RoBERTa](https://huggingface.co/Gladiator/roberta-large_ner_conll2003)<sub>large</sub> | 96.57 |
| T5 | - |
| T5-Enc<sub>large</sub> | 95.49 |

**Important:** Those results are obtained by a single run, for those datasets with very few examples the performance might change drastically.
