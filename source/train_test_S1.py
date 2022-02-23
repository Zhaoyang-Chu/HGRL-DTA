import os, json, torch
import numpy as np
from collections import OrderedDict

from model import ConvNet, FirstVariantOfConvNet, SecondVariantOfConvNet, ThirdVariantOfConvNet, Predictor
from metrics import model_evaluate
from GraphInput import getAffinityGraph, getDrugMolecularGraph, getTargetMolecularGraph
from utils import argparser, DTADataset, GraphDataset, collate, getLinkEmbeddings, predicting, read_data, train


def create_dataset_for_train_test(affinity, dataset, fold, weighted, drug_aff_k, target_aff_k):
    # load dataset
    dataset_path = 'data/' + dataset + '/'

    train_fold_origin = json.load(open(dataset_path + 'S1_train_set.txt'))
    train_folds = []
    for i in range(len(train_fold_origin)):
        if i != fold:
            train_folds += train_fold_origin[i]
    test_fold = json.load(open(dataset_path + 'S1_test_set.txt')) if fold == -100 else train_fold_origin[fold]

    rows, cols = np.where(np.isnan(affinity) == False)

    train_rows, train_cols = rows[train_folds], cols[train_folds]
    train_Y = affinity[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = rows[test_fold], cols[test_fold]
    test_Y = affinity[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    adj = np.zeros_like(affinity)
    adj[train_rows, train_cols] = train_Y
    affinity_graph = getAffinityGraph(dataset, adj, weighted, drug_aff_k, target_aff_k)

    return train_dataset, test_dataset, affinity_graph


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
    if not FLAGS.weighted:
        model_name += "-noweight"
    if fold != -100:
        model_name += f"-{fold}"

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Epochs:", NUM_EPOCHS)
    print("Learning rate:", LR)
    print("Model name:", model_name)
    print("Train and test") if fold == -100 else print("Fold of 5-CV:", fold)
    
    if os.path.exists(f"models/architecture/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/predictor/{dataset}/S1/cross_validation/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/cross_validation/")
    if os.path.exists(f"models/architecture/{dataset}/S1/test/") is False:
        os.makedirs(f"models/architecture/{dataset}/S1/test/")
    if os.path.exists(f"models/predictor/{dataset}/S1/test/") is False:
        os.makedirs(f"models/predictor/{dataset}/S1/test/")

    print("create dataset ...")
    affinity = read_data(dataset)
    train_data, test_data, affinity_graph = create_dataset_for_train_test(affinity, dataset, fold, FLAGS.weighted, FLAGS.drug_aff_k, FLAGS.target_aff_k)
    print("create train_loader and test_loader ...")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")
    drug_graphs_dict = getDrugMolecularGraph(json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    target_graphs_dict = getTargetMolecularGraph(json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict), dataset)
    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node1s)  # if memory is not enough, turn down the batch_size, e.g., batch_size=30
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node2s)  # if memory is not enough, turn down the batch_size, e.g., batch_size=100

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture(ag_init_dim=affinity_graph.num_node1s + affinity_graph.num_node2s + 2, affinity_dropout_rate=FLAGS.dropedge_rate)
    architecture.to(device)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)

    if fold != -100:
        best_result = [1000]
    print("start training ...")
    for epoch in range(NUM_EPOCHS):
        train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch + 1, TRAIN_BATCH_SIZE, affinity_graph)
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph)
        result = model_evaluate(G, P, dataset)
        if fold != -100 and result[0] < best_result[0]:
            best_result = result
            checkpoint_dir = f"models/architecture/{dataset}/S1/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)
            
            checkpoint_dir = f"models/predictor/{dataset}/S1/cross_validation/"
            checkpoint_path = checkpoint_dir + model_name + ".pkl"
            torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

    if fold == -100:
        checkpoint_dir = f"models/architecture/{dataset}/S1/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(architecture.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        checkpoint_dir = f"models/predictor/{dataset}/S1/test/"
        checkpoint_path = checkpoint_dir + model_name + ".pkl"
        torch.save(predictor.state_dict(), checkpoint_path, _use_new_zipfile_serialization=False)

        print('\npredicting for test data')
        G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph)
        result = model_evaluate(G, P, dataset)
        print("reslut:", result)
    else:
        print(f"\nbest result for fold {fold} of cross validation:")
        print("reslut:", best_result)


def test():
    
    FLAGS = argparser()
    
    dataset = FLAGS.dataset
    cuda_name = f'cuda:{FLAGS.cuda_id}'
    TRAIN_BATCH_SIZE = FLAGS.batch_size
    TEST_BATCH_SIZE = FLAGS.batch_size
    Architecture = [ConvNet, FirstVariantOfConvNet, SecondVariantOfConvNet, ThirdVariantOfConvNet][FLAGS.model]
    model_name = Architecture.__name__

    print("Dataset:", dataset)
    print("Cuda name:", cuda_name)
    print("Model name:", model_name)

    print("create dataset ...")
    affinity = read_data(dataset)
    train_data, test_data, affinity_graph = create_dataset_for_train_test(affinity, dataset)
    print("create train_loader and test_loader ...")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate)

    print("create drug_graphs_dict and target_graphs_dict ...")
    drug_graphs_dict = getDrugMolecularGraph(json.load(open(f'data/{dataset}/drugs.txt'), object_pairs_hook=OrderedDict))
    target_graphs_dict = getTargetMolecularGraph(json.load(open(f'data/{dataset}/targets.txt'), object_pairs_hook=OrderedDict), dataset)
    print("create drug_graphs_DataLoader and target_graphs_DataLoader ...")
    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node1s)
    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate, batch_size=affinity_graph.num_node2s)

    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    architecture = Architecture(ag_init_dim=affinity_graph.num_node1s + affinity_graph.num_node2s + 2)
    architecture.to(device)

    predictor = Predictor(embedding_dim=architecture.output_dim)
    predictor.to(device)

    architecture_ckpt = torch.load(f"models/architecture/{dataset}/S1/test/{model_name}.pkl")
    architecture.load_state_dict(architecture_ckpt)

    predictor_ckpt = torch.load(f"models/predictor/{dataset}/S1/test/{model_name}.pkl")
    predictor.load_state_dict(predictor_ckpt)

    print('predicting for test data')
    G, P = predicting(architecture, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph)
    result = model_evaluate(G, P, dataset)
    print("reslut:", result)

    print('Get embeddings of links on test set...')
    link_embeddings = getLinkEmbeddings(architecture, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph)
    print(link_embeddings.shape)
    np.savetxt(f"embeddings/{dataset}/MIGNN/test_link_embeddings.txt", link_embeddings, delimiter=",")


if __name__ == '__main__':
    train_test()
    # test()
