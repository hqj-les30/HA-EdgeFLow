# HA-EdgeFLow
This repository provides an implementation of **HA-EdgeFLow**, an aggregation strategy for federated learning method derived from the theoretical analysis proposed in our paper:

> **EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks**  
> Yuchen Shi, Qijun Hou, Pingyi Fan, Khaled B. Letaief  
> *IEEE Global Communications Conference (GLOBECOM), 2025*

---

## Overview

HA-EdgeFLow is a **client selection** framework designed for **federated learning with a mobile server**.  

Based on the theoretical analysis in the paper, the core idea of EdgeFLow is to **prioritize client models whose gradients are closer to a proxy global gradient**, which helps stabilize training and improve convergence under non-IID data distributions.


## Code Structure

- `main.py`: Entry for running experiments
- `data/`: Dataset original files
- `result/`: Default path to store results
- `server.py`: Definition of the server and the selection algorithm

---

## Requirements

- Python >= 3.8
- PyTorch (cuda)
- torchvision
- numpy

(Please install dependencies according to your environment.)

---

## Usage

### Running an Experiment

The main entry point of the code is `main.py`.  
A typical command is:

```bash
python main.py \
  --dataset cifar10 \
  --n_clients 100 \
  --global_epoch 1000 \
  --cluster edgeflow \
  --cluster_size 10 \
  --l 2
```

## Citation

If you find this code useful in your research, please cite:
```
@INPROCEEDINGS{edgeflow,
  author    = {Shi, Yuchen and Hou, Qijun and Fan, Pingyi and Letaief, Khaled B.},
  booktitle = {2025 IEEE Global Communications Conference},
  title     = {EdgeFLow: Serverless Federated Learning via Sequential Model Migration in Edge Networks},
  year      = {2025},
  note      = {Accepted and presented}
}
```
