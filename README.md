# generative-ai

51M parameters model trained on 11B tokens.

from Sep 6 17:40-
15h 8:40

## Prerequisites

Tested on:

Python 3.10.12
Poetry 1.6.1
NVIDIA V100 GPU
CUDA 11.8

## Getting Started

```sh
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/jupyter/.local/bin:$PATH"
poetry install
poetry run python generative_ai/scripts/train.py
```
