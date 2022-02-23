# Normalize Similarities
import numpy as np


def minMaxNormalize(Y, Y_min=None, Y_max=None):
    if Y_min is None:
        Y_min = np.min(Y)
    if Y_max is None:
        Y_max = np.max(Y)
    normalize_Y = (Y - Y_min) / (Y_max - Y_min)
    return normalize_Y


davis_drug_sim = np.loadtxt("davis/drug-drug_similarities_2D.txt", delimiter=" ", usecols=range(68))  # from https://github.com/hkmztrk/DeepDTA/blob/master/data/davis/drug-drug_similarities_2D.txt or https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/drug-drug_similarities_2D.txt
davis_target_sim = np.loadtxt("davis/target-target_similarities_WS_normalized.txt", delimiter=" ", usecols=range(442))  # from http://staff.cs.utu.fi/~aatapa/data/DrugTarget/target-target_similarities_WS_normalized.txt
kiba_drug_sim = np.loadtxt("kiba/kiba_drug_sim.txt", delimiter="	", usecols=range(2111))  # from https://github.com/hkmztrk/DeepDTA/blob/master/data/kiba/kiba_drug_sim.txt or https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/kiba_drug_sim.txt
kiba_target_sim = np.loadtxt("kiba/kiba_target_sim.txt", delimiter="	", usecols=range(229))  # from https://github.com/hkmztrk/DeepDTA/blob/master/data/kiba/kiba_target_sim.txt or https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/kiba_target_sim.txt

print(davis_drug_sim)
print(davis_target_sim)
print(kiba_drug_sim)
print(kiba_target_sim)

davis_drug_sim = davis_drug_sim - np.eye(davis_drug_sim.shape[0])
davis_target_sim = minMaxNormalize(davis_target_sim, 0, 100) - np.eye(davis_target_sim.shape[0])
kiba_drug_sim = kiba_drug_sim - np.eye(kiba_drug_sim.shape[0])
kiba_target_sim = kiba_target_sim - np.eye(kiba_target_sim.shape[0])

print(davis_drug_sim)
print(davis_target_sim)
print(kiba_drug_sim)
print(kiba_target_sim)

np.savetxt("davis/drug-drug-sim.txt", davis_drug_sim, delimiter=",")
np.savetxt("davis/target-target-sim.txt", davis_target_sim, delimiter=",")
np.savetxt("kiba/drug-drug-sim.txt", kiba_drug_sim, delimiter=",", fmt="%.6f")
np.savetxt("kiba/target-target-sim.txt", kiba_target_sim, delimiter=",")
