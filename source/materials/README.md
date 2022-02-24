# Preprocess materials

## 1 Description of Raw Materials
Download the following raw files of two benchmark datasets (i.e., Davis and KIBA) from https://github.com/hkmztrk/DeepDTA/tree/master/data:
- `ligands_can.txt`: drug SMILES strings
- `proteins.txt`: target protein sequences
- `Y`: the drug-target affinity matrix
- `folds/train_fold_setting1.txt`: the training set of setting S1
- `folds/test_fold_setting1.txt`: the test set of setting S1

## 2 Normalize Similarities
1. Download similarity matrix files of drugs and targets.
2. Normalize similarities into the range [0, 1] using min-max normalization. See `materials/normalize_sim.py` for details.

## 3 Delete Protein P78527
Delete one large protein target, called P78527, and its related affinities and similarities from the KIBA dataset because of the memory limitation of our machine. See `materials/kiba/delete.py` for details.

## 4 Move
Move files from `materials/` to `data/`. See `data/README.md` for details.
