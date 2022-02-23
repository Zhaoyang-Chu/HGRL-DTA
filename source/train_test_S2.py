import os, json, torch
import numpy as np
from collections import OrderedDict

from model import ConvNet, FirstVariantOfConvNet, SecondVariantOfConvNet, ThirdVariantOfConvNet, Predictor
from metrics import model_evaluate
from GraphInput import getAffinityGraph, getDrugMolecularGraph, getTargetMolecularGraph
from utils import argparser, DTADataset, GraphDataset, collate, predicting, read_data, train


def create_dataset_for_train_test(affinity, dataset, fold, weighted, drug_aff_k, target_aff_k, drug_sim_k):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    drug_train_fold_origin = json.load(open(dataset_path + 'S2_train_set.txt'))
    drug_train_folds = []
    for i in range(len(drug_train_fold_origin)):
        if i != fold:
            drug_train_folds += drug_train_fold_origin[i]
    drug_test_fold = json.load(open(dataset_path + 'S2_test_set.txt')) if fold == -100 else drug_train_fold_origin[fold]
    drug_mask_fold = json.load(open(dataset_path + 'S2_mask_set.txt')) if fold == -100 else json.load(open(dataset_path + 'S2_test_set.txt')) + json.load(open(dataset_path + 'S2_mask_set.txt'))

    # train set and test set
    train_affinity = affinity[drug_train_folds, :]
    test_affinity = affinity[drug_test_fold, :]

    train_rows, train_cols = np.where(np.isnan(train_affinity) == False)
    train_Y = train_affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = np.where(np.isnan(test_affinity) == False)
    test_Y = test_affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    train_affinity[np.isnan(train_affinity) == True] = 0
    affinity_graph = getAffinityGraph(dataset, train_affinity, weighted, drug_aff_k, target_aff_k)

    # drug molecular graphs
    drugs = json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict)
    drug_keys = np.array(list(drugs.keys()))
    drug_values = np.array(list(drugs.values()))
    train_drug_keys = drug_keys[drug_train_folds]
    train_drug_values = drug_values[drug_train_folds]
    train_drugs = dict(zip(train_drug_keys, train_drug_values))
    test_drug_keys = drug_keys[drug_test_fold]
    test_drug_values = drug_values[drug_test_fold]
    test_drugs = dict(zip(test_drug_keys, test_drug_values))

    train_drug_graphs_dict = getDrugMolecularGraph(train_drugs)
    test_drug_graphs_dict = getDrugMolecularGraph(test_drugs)

    # drug map
    drug_sim = np.loadtxt(f"data/{dataset}/drug-drug-sim.txt", delimiter=",")
    drug_test_train_sim = drug_sim[drug_test_fold, :]
    drug_test_train_sim[:, drug_test_fold + drug_mask_fold] = -1

    drug_count = affinity.shape[0]
    drug_train_count = len(drug_train_folds)
    drug_test_train_map = np.argpartition(drug_test_train_sim, -drug_sim_k, 1)[:, -drug_sim_k:]
    drug_train_map = np.full(drug_count, -1)
    drug_train_map[drug_train_folds] = np.arange(drug_train_count)
    drug_test_map = drug_train_map[drug_test_train_map]

    drug_test_map_weight = drug_test_train_sim[np.tile(np.expand_dims(np.arange(drug_test_train_sim.shape[0]), 0), (drug_sim_k, 1)).transpose(), drug_test_train_map]
    drug_test_map_weight_sum = np.expand_dims(np.sum(drug_test_map_weight, axis=1), axis=1)
    drug_test_map_weight_norm = np.expand_dims(drug_test_map_weight / drug_test_map_weight_sum, axis=2)

    target_graphs_dict = getTargetMolecularGraph(json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict), dataset)

    return train_dataset, test_dataset, affinity_graph, train_drug_graphs_dict, test_drug_graphs_dict, target_graphs_dict, drug_count, drug_test_map, drug_test_map_weight_norm


def train_test():

    FLAGS = argparser()

    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    NUM_EPOCHS = FLAGS.num_epochs
    LR = FLAGS.lr
    Architecture = [ConvNet, FirstVariantOfConvNet, SecondVariantOfConvNet, ThirdVariantOfConvNet][FLAGS.model]
    model_name = Architecture.__name__
    fold = FLAGS.fold
    if fold != -100:
        model_name += f"-{FLAGS.drug_sim_k}-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    
    if os.path.exists(f"models/architecture/{dataset}/S2/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S2/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S2/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S2/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S2/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S2/test/")
    if os.path.exists(f"models/predictor/{dataset}/S2/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S2/test/")

    print("create dataset ...")
    affinity = read_data(dataset)
    train_data, test_data, affinity_graph, train_drug_graphs_dict, test_drug_graphs_dict, target_graphs_dict, drug_count, drug_test_map, drug_test_map_weight_norm = \
        create_dataset_for_train_test(affinity, dataset, fold, FLAGS.weighted, FLAGS.drug_aff_k, FLAGS.target_aff_k, FLAGS.drug_sim_k)
    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    train_drug_graphs_Data = GraphDataset(graphs_dict=train_drug_graphs_dict, dttype="drug")
    train_drug_graphs_DataLoader = torch.utils.data.DataLoader(train_drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node1s)
    test_drug_graphs_Data = GraphDataset(graphs_dict=test_drug_graphs_dict, dttype="drug")
    test_drug_graphs_DataLoader = torch.utils.data.DataLoader(test_drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=drug_count - affinity_graph.num_node1s)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node2s)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture(ag_init_dim=affinity_graph.num_node1s + affinity_graph.num_node2s + 2, skip=True)
    architecture.to(device)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)

    if fold != -100:
        best_result = [1000]
    print("start training ...")
    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, train_drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE, affinity_graph)
        G, P = predicting(architecture, predictor, device, test_loader, test_drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_map=drug_test_map, drug_map_weight=torch.FloatTensor(drug_test_map_weight_norm).to(device))
        result = model_evaluate(G, P, dataset)
        if fold != -100 and result[0] < best_result[0]:
            best_result = result
            checkpoint_dir = f"models/architecture/{dataset}/S2/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
            
            checkpoint_dir = f"models/predictor/{dataset}/S2/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

    if fold == -100:
        checkpoint_dir = f"models/architecture/{dataset}/S2/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_dir = f"models/predictor/{dataset}/S2/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')
        G, P = predicting(architecture, predictor, device, test_loader, test_drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_map=drug_test_map, drug_map_weight=torch.FloatTensor(drug_test_map_weight_norm).to(device))
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


if __name__ == '__main__':
    train_test()
