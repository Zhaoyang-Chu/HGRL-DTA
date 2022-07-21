# HGRL-DTA
This repository contains a PyTorch implementation of the paper "Hierarchical Graph Representation Learning for the Prediction of Drug-Target Binding Affinity".

<img src="./Framework.jpg" style="zoom: 100%;"/>

## Overview of Source Codes
- `materials/` contains raw materials of the Davis dataset and the KIBA dataset.
- `data/` contains the input data of our model.
- `metrics.py`: contains the evaluation metrics used in our experiments (i.e., MSE, CI, $r_m^2$, Pearson, and AUPR).
- `GraphInput.py`: contains the construction processes of the affinity graph, the drug molecule graph and the target molecule graph.
- `model.py`: contains our HGRL-DTA model and its variants.
- `train_test_S1.py`: contains the training and testing processes under the **S1** setting.
- `train_test_S2.py`: contains the training and testing processes under the **S2** setting.
- `train_test_S3.py`: contains the training and testing processes under the **S3** setting.
- `train_test_S4.py`: contains the training and testing processes under the **S4** setting.
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

## Runing

### Data Preparation
Prepare target molecule graphs, please refer to [Prepare Target Molecule Graphs](https://github.com/Zhaoyang-Chu/HGRL-DTA/tree/main/source/data#2-prepare-for-target-molecule-graphs).

### Setting S1

#### Cross Validation
Cross validation our model on the Davis dataset:
```shell
python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 ---fold 0
python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 ---fold 1
python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 ---fold 2
python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 ---fold 3
python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 ---fold 4
```
Cross validation under other experimental settings is similar.

#### Train and Test
- Train and test our model on the Davis dataset:
    ```shell
    python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2
    ```
- Train and test our model on the KIBA dataset:
    ```shell
    python train_test_S1.py --dataset kiba --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --drug_aff_k 40 --target_aff_k 150
    ```

#### Ablation Study
Ablation study on the Davis dataset:
- HGRL-DTA (w/o CAG):
    ```shell
    python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 2 --dropedge_rate 0.2
    ```
- HGRL-DTA (w/o FMG):
    ```shell
    python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 1 --dropedge_rate 0.2
    ```
- HGRL-DTA (w/o WA):
    ```shell
    python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 0 --weighted --dropedge_rate 0.2
    ```
- HGRL-DTA-L:
    ```shell
    python train_test_S1.py --dataset davis --cuda_id 0 --num_epochs 2000 --batch_size 512 --lr 0.0005 --model 3 --dropedge_rate 0.2
    ```
Ablation study on the KIBA dataset is similar.

### Setting S2
- Train and test our model on the Davis dataset:
    ```shell
    python train_test_S2.py --dataset davis --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --drug_sim_k 2 --skip
    ```
- Train and test our model on the KIBA dataset:
    ```shell
    python train_test_S2.py --dataset kiba --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --drug_aff_k 40 --target_aff_k 90 --drug_sim_k 2 --skip
    ```
To prevent over-fitting, the same setting (i.e, 200 epochs, 512 batch size and 0.0005 learning rate) is used when runing other compared methods.

### Setting S3
- Train and test our model on the Davis dataset:
    ```shell
    python train_test_S3.py --dataset davis --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --target_sim_k 7 --skip
    ```
- Train and test our model on the KIBA dataset:
    ```shell
    python train_test_S3.py --dataset kiba --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --drug_aff_k 40 --target_aff_k 150 --target_sim_k 7 --skip
    ```
To prevent over-fitting, the same setting (i.e, 200 epochs, 512 batch size and 0.0005 learning rate) is used when runing other compared methods.

### Setting S4
- Train and test our model on the Davis dataset:
    ```shell
    python train_test_S4.py --dataset davis --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --drug_sim_k 2 --target_sim_k 7 --skip
    ```
- Train and test our model on the KIBA dataset:
    ```shell
    python train_test_S4.py --dataset kiba --cuda_id 0 --num_epochs 200 --batch_size 512 --lr 0.0005 --model 0 --dropedge_rate 0.2 --drug_aff_k 40 --target_aff_k 90 --drug_sim_k 2 --target_sim_k 7 --skip
    ```
To prevent over-fitting, the same setting (i.e, 200 epochs, 512 batch size and 0.0005 learning rate) is used when runing other compared methods.
