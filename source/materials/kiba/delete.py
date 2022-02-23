# Delete Protein P78527
import pickle, json, random
from collections import OrderedDict
import numpy as np


# affinity
def affinity():
    affinity = pickle.load(open('Y', 'rb'), encoding='latin1')
    affinity = np.asarray(affinity)
    print(affinity.shape)

    affinity = np.delete(affinity, 128, axis=1)

    pickle.dump(affinity, open('Y_new', 'wb'))

    affinity = pickle.load(open('Y_new', 'rb'), encoding='latin1')
    affinity = np.asarray(affinity)
    print(affinity.shape)


# proteins
def proteins():
    proteins = json.load(open('proteins.txt'), object_pairs_hook=OrderedDict)
    print(list(proteins.keys())[128])

    proteins.pop("P78527")
    print(list(proteins.keys())[128])

    json.dump(proteins, open('proteins_new.txt', "w"))

    proteins = json.load(open('proteins_new.txt'), object_pairs_hook=OrderedDict)
    print(list(proteins.keys())[128])


# folds
def folds():
    affinity = pickle.load(open('Y_new', 'rb'), encoding='latin1')
    affinity = np.asarray(affinity)
    rows, cols = np.where(np.isnan(affinity) == False)
    print(len(rows))

    num = len(rows)
    k_folds = 6
    random_index = np.arange(num)
    random.seed(1)
    random.shuffle(random_index)

    CV_size = int(num / k_folds)
    print(CV_size)
    all_folds = np.array(random_index[:num - num % k_folds]).reshape(k_folds, CV_size).tolist()
    all_folds[k_folds - 1] = all_folds[k_folds - 1] + list(random_index[num - num % k_folds:])

    fold = 5
    train_folds = []
    test_fold = [int(i) for i in all_folds[fold]]
    for i in range(k_folds):
        if i != fold:
            train_folds.append(all_folds[i])
    
    json.dump(train_folds, open('folds/train_fold_setting1_new.txt', "w"))
    json.dump(test_fold, open('folds/test_fold_setting1_new.txt', "w"))

    train_fold_origin = json.load(open('folds/train_fold_setting1_new.txt'))
    test_fold = json.load(open('folds/test_fold_setting1_new.txt'))


def sim():
    sim = np.loadtxt("target-target-sim.txt", delimiter=",")
    print(sim)
    print(sim.shape)

    sim = np.delete(np.delete(sim, 128, axis=1), 128, axis=0)
    print(sim)
    print(sim.shape)

    np.savetxt("target-target-sim_new.txt", sim, delimiter=",")
    sim = np.loadtxt("target-target-sim_new.txt", delimiter=",")
    print(sim)
    print(sim.shape)


if __name__ == "__main__":
    affinity()
    proteins()
    folds()
    sim()