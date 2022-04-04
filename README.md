# Task Meta Transfer (TMT)
This repo covers the implementation of **[Task Meta-Transfer from Limited Parallel Labels](https://meta-learn.github.io/2020/papers/15_paper.pdf)** presented at 4th Meta-Learning workshop, NeurIPS 2020. [Video](https://slideslive.com/38941946/task-metatransfer-from-limited-parallel-labels) and [Poster](https://meta-learn.github.io/2020/papers/15_poster.png) are available.

<img src="figures/overview.png" width="400">

```bibtex
@article{Jian2020TMT,
    author = {Yiren Jian, Karim Ahmed and Lorenzo Torresani},
    title = {Task Meta-Transfer from Limited Parallel Labels},
    journal = {4th Meta-Learning workshop, NeurIPS 2020},
    year = {2020}
}
```

## Requirements
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.6, PyTorch 1.4.0, and CUDA 10.1. You will need at least 11GB VRAM (e.g. Nvidia RTX-2080Ti) for running full experiments in this repo.

## Data Preparation
We use the pre-processed NYUv2 and CityScapes from [mtan](https://github.com/lorenmt/mtan). E.g., we put the unzip files in `/home/yiren/datasets/nyuv2`.

## Training
<img src="figures/method.png" width="400">

Training with Task Meta-Transfer (TMT):
```
python nyu_TMT.py \
    --rank_projection \
    --update_freq 5 \
    --rank 5 \
    --model_size M \
    --data_size S \
    --dataroot /home/yiren/datasets/nyuv2 \
    --target_task semantic \
    --auxi_task depth
```
Setting `--target_task` and `--auxi_task` from `semantic/depth/normal` for the primary target task and auxiliary task.

Running all experiments by the following commands
```
#### VGG
python nyu_TMT.py --arch deconv_3d --rank_projection --rank 5   --update_freq 5 --target_task semantic --auxi_task depth
python nyu_TMT.py --arch deconv_3d --rank_projection --rank 100 --update_freq 8 --target_task semantic --auxi_task normal

python nyu_TMT.py --arch deconv_3d --rank_projection --rank 5   --update_freq 4 --target_task depth --auxi_task semantic
python nyu_TMT.py --arch deconv_3d --rank_projection --rank 40  --update_freq 5 --target_task depth --auxi_task normal

python nyu_TMT.py --arch deconv_3d --rank_projection --rank 40  --update_freq 2 --target_task normal --auxi_task semantic
python nyu_TMT.py --arch deconv_3d --rank_projection --rank 40  --update_freq 9 --target_task normal --auxi_task depth

#### resnet
python nyu_TMT.py --arch resnet_3d --rank_projection --rank 60   --update_freq 10 --target_task semantic --auxi_task depth
python nyu_TMT.py --arch resnet_3d --rank_projection --rank 60   --update_freq 9  --target_task semantic --auxi_task normal

python nyu_TMT.py --arch resnet_3d --rank_projection --rank 80  --update_freq 3  --target_task depth --auxi_task semantic
python nyu_TMT.py --arch resnet_3d --rank_projection --rank 60  --update_freq 9  --target_task depth --auxi_task normal

python nyu_TMT.py --arch resnet_3d --rank_projection --rank 40  --update_freq 5  --target_task normal --auxi_task semantic
python nyu_TMT.py --arch resnet_3d --rank_projection --rank 40  --update_freq 9  --target_task normal --auxi_task depth
```

Baseline single task training (without auxiliary task learning):
```
python nyu_baseline.py
```

## Contacts
For any questions, please contact authors.

## Acknowledgment
Part of our code is borrowed from [mtan](https://github.com/lorenmt/mtan) and [MAXL](https://github.com/lorenmt/maxl).
