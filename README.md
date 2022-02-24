# HGRL-DTA
This repository contains a PyTorch implementation of the paper "Hierarchical Graph Representation Learning for the Prediction of Drug-Target Binding Affinity".

<img src="./Framework.jpg" style="zoom: 100%;" />

## Overview of Source Codes
- `materials/` contains raw materials of the Davis dataset and the KIBA dataset.
- `data/` contains the input data of our model.
- `metrics.py`: contains the evaluation metrics used in our experiments (i.e., MSE, CI, $r_m^2$, Pearson and AUPR).
- `GraphInput.py`: contains the construction processes of the affinity graph, the drug molecular graph and the target molecular graph.
- `model.py`: contains our HGRL-DTA model and its variants.
- `train_test_S1.py`: contains the training and testing processes under setting S1.
- `train_test_S2.py`: contains the training and testing processes under setting S2.
- `train_test_S3.py`: contains the training and testing processes under setting S3.
- `train_test_S4.py`: contains the training and testing processes under setting S4.
- `utils.py`: contains utility functions.

## Dependencies
- numpy == 1.17.4
- scikit-learn == 0.22.2
- rdkit == 2017.09.1
- networkx == 2.5
- torch == 1.4.0
- torch-geometric == 1.7.0
- lifelines == 0.25.6
- argparse == 1.4.0

## Data Preparation

## Runing

### Setting S1

#### Train and Test

#### Ablation Study

### Setting S2

#### Train and Test

#### Parameter Analysis

### Setting S3

#### Train and Test

#### Parameter Analysis

### Setting S4

#### Train and Test
