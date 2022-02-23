from json.tool import main
import pickle, sys
import numpy as np


def read_data(dataset):
    dataset_path = dataset + '/'
    affinity = pickle.load(open(dataset_path + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
        affinity = np.asarray(affinity) - 5
    else:
        affinity = np.asarray(affinity)
        affinity[np.isnan(affinity)] = 0
    return affinity


def degrees(affinity):
    print("Drug degree:", np.mean(np.sum(affinity != 0, axis=1)), np.max(np.sum(affinity != 0, axis=1)))
    print("Target degree:", np.mean(np.sum(affinity != 0, axis=0)), np.max(np.sum(affinity != 0, axis=0)))
    
    
if __name__ == '__main__':
    affinity = read_data(sys.argv[1])
    degrees(affinity)