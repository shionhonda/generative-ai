# generative-ai

This repository aims to cover minimal codes for generative models for texts and images. They basically depend on PyTorch 2.0, no HugginFace transformers.

As a first step, I included the code to train a 51M-parameter language model with 11B tokens.

from Sep 6 17:40 -> 18h -> 20:40

## Prerequisites

Tested on:

Python 3.10.12
Poetry 1.6.1
NVIDIA V100 GPU
CUDA 11.8

## Getting Started

To create a tokenizer, run:

```sh
poetry run python generative_ai/scripts/create_tokenizer.py
```

To launch training, run:

```sh
poetry run python generative_ai/scripts/train.py
```

![](fig/loss.png)

To generate sentences with pretrained model, run:

```sh
poetry run python generative_ai/scripts/generate.py
```
