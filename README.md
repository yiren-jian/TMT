# Task Meta Transfer
This repo covers the implementation of "Task Meta-Transfer from Limited Parallel Labels".

## Requirements
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, PyTorch 1.4.0, and CUDA 10.1. You will need at least 11GB VRAM (e.g. Nvidia RTX-2080Ti) for running full experiments in this repo.

## Data Preparation
We use the pre-processed NYUv2 from [mtan](https://github.com/lorenmt/mtan).

## Training
Training with Task Meta-Transfer:
```
python TMT.py --rank 5 --model_size M --data_size S --dataroot /home/yiren/datasets/nyuv2
```

Baseline single task training:
```
python nyu_base.py
```

## Contacts
For any questions, please contact authors.

## Acknowledgment
Part of our code is borrowed from [mtan](https://github.com/lorenmt/mtan).
