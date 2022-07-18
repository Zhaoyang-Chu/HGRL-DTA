import pickle, argparse
import numpy as np
from itertools import chain
import torch
import torch.nn as nn
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, Batch


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset for use', default='davis')
    parser.add_argument('--cuda_id', type=int, help='Cuda for use', default=0)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train', default=2000)  # num_epochs = 200, when conducting the S2, S3 and S4 experiments
    parser.add_argument('--batch_size', type=int, help='Batch size of dataset', default=512)
    parser.add_argument('--lr', type=float, help='Initial learning rate to train', default=0.0005)
    parser.add_argument('--model', type=int, help='Model id', default=0)
    parser.add_argument('--fold', type=int, help='Fold of 5-CV', default=-100)
    parser.add_argument('--weighted', help='Whether affinity graph is weighted', action='store_false')
    parser.add_argument('--dropedge_rate', type=float, help='Rate of edge dropout', default=0.2)
    parser.add_argument('--drug_sim_k', type=int, help='Similarity topk of drug', default=2)
    parser.add_argument('--target_sim_k', type=int, help='Similarity topk of target', default=7)
    parser.add_argument('--drug_aff_k', type=int, help='Affinity topk of drug', default=40)
    parser.add_argument('--target_aff_k', type=int, help='Affinity topk of target', default=150)  # target_aff_k = 90, when conducting the S2 and S4 experiments on the KIBA dataset
    parser.add_argument('--skip', help='Whether the skip connection operation is used', action='store_true')
    FLAGS, unparsed = parser.parse_known_args()

    return FLAGS


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, drug_ids=None, target_ids=None, y=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)
        self.process(drug_ids, target_ids, y)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, drug_ids, target_ids, y):
        data_list = []
        for i in range(len(drug_ids)):
            DTA = DATA.Data(drug_id=torch.IntTensor([drug_ids[i]]), target_id=torch.IntTensor([target_ids[i]]), y=torch.FloatTensor([y[i]]))
            data_list.append(DTA)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    
class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(architecture, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, LR, epoch, TRAIN_BATCH_SIZE, affinity_graph):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    architecture.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, chain(architecture.parameters(), predictor.parameters())), lr=LR, weight_decay=0)
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        drug_embedding, target_embedding = architecture(affinity_graph, drug_graph_batchs, target_graph_batchs)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * TRAIN_BATCH_SIZE, len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()
            ))


def predicting(architecture, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
    architecture.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        for data in loader:
            drug_embedding, target_embedding = architecture(
                affinity_graph, drug_graph_batchs, target_graph_batchs, 
                drug_map=drug_map, drug_map_weight=drug_map_weight, target_map=target_map, target_map_weight=target_map_weight
            )
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def getLinkEmbeddings(architecture, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
    architecture.eval()
    predictor.eval()
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        link_embeddings_batch_list = []
        for data in loader:
            drug_embedding, target_embedding = architecture(
                affinity_graph, drug_graph_batchs, target_graph_batchs, 
                drug_map=drug_map, drug_map_weight=drug_map_weight, target_map=target_map, target_map_weight=target_map_weight
            )
            _, link_embeddings_batch = predictor(data.to(device), drug_embedding, target_embedding)
            link_embeddings_batch_list.append(link_embeddings_batch.cpu().numpy())
    link_embeddings = np.concatenate(link_embeddings_batch_list, axis=0)
    return link_embeddings


def getEmbeddings(architecture, device, drug_graphs_DataLoader, target_graphs_DataLoader, affinity_graph):
    architecture.eval()
    affinity_graph.to(device)  # affinity graph
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))  # drug graphs
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))  # target graphs
    with torch.no_grad():
        drug_embedding, target_embedding = architecture(affinity_graph, drug_graph_batchs, target_graph_batchs)
    return drug_embedding.cpu().numpy(), target_embedding.cpu().numpy()


def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch


def read_data(dataset):
    dataset_path = 'data/' + dataset + '/'
    affinity = pickle.load(open(dataset_path + 'affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    return affinity
