import torch
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import GCNConv, DenseGCNConv, global_mean_pool as gep
from torch_geometric.utils import dropout_adj


vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim)
}


class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        emdeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            emdeddings.append(output)
        return emdeddings
    

class DenseGCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[], supplement_mode=None):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = DenseGCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj, supplement_x=None):
        output = x
        emdeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                supplement_x = torch.unsqueeze(supplement_x, 0)
                output = self.supplement_func(output, supplement_x)
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            emdeddings.append(torch.squeeze(output, dim=0))
        return emdeddings
    

class GCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[], supplement_mode=None):
        super(GCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None):
        output = x
        emdeddings = []
        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                output = self.supplement_func(output, supplement_x)
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            emdeddings.append(gep(output, batch))
        return emdeddings


class DenseGCNModel(torch.nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0, supplement_mode=None):
        super(DenseGCNModel, self).__init__()
        print('DenseGCNModel Loaded')

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=range(self.num_layers), dropout_layers_index=range(self.num_layers), supplement_mode=supplement_mode)

    def forward(self, graph, supplement_x=None):
        xs, adj, num_node1s, num_node2s = graph.x, graph.adj, graph.num_node1s, graph.num_node2s
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs], p=self.edge_dropout_rate, force_undirected=True, num_nodes=num_node1s + num_node2s, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout, supplement_x=supplement_x)

        return embeddings


class GCNModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None):
        super(GCNModel, self).__init__()
        print('GCNModel Loaded')

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=range(self.num_layers), supplement_mode=supplement_mode)

    def forward(self, graph_batchs, supplement_x=None):

        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x', supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs
            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch, supplement_x=graph.supplement_x[graph.batch.int().cpu().numpy()]), graph_batchs))
        else:
            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        embeddings = []
        for i in range(self.num_layers):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings


class ConvNet(torch.nn.Module):
    def __init__(self, ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, affinity_dropout_rate=0.2, skip=False, embedding_dim=128, integration_mode="combination4"):
        super(ConvNet, self).__init__()
        print('ConvNet Loaded')

        affinity_graph_dims = [ag_init_dim, 512, 256]

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_transform_dims = [affinity_graph_dims[-1], 1024, drug_graph_dims[1]]
        target_transform_dims = [affinity_graph_dims[-1], 1024, target_graph_dims[1]]

        self.skip = skip
        if not skip:
            drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]
        else:
            drug_output_dims = [drug_graph_dims[-1] + drug_transform_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1] + target_transform_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, affinity_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

        self.drug_graph_conv = GCNModel(drug_graph_dims, supplement_mode=integration_mode)
        self.target_graph_conv = GCNModel(target_graph_dims, supplement_mode=integration_mode)
        
        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):

        num_node1s, num_node2s = affinity_graph.num_node1s, affinity_graph.num_node2s

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1]

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs, supplement_x=drug_transform_embedding)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs, supplement_x=target_transform_embedding)[-1]

        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[-1]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[-1]
        else:
            drug_output_embedding = self.drug_output_linear(torch.cat((drug_graph_embedding, drug_transform_embedding), 1))[-1]
            target_output_embedding = self.target_output_linear(torch.cat((target_graph_embedding, target_transform_embedding), 1))[-1]

        return drug_output_embedding, target_output_embedding


class FirstVariantOfConvNet(torch.nn.Module):
    def __init__(self, ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, affinity_dropout_rate=0.2, skip=False, embedding_dim=128, integration_mode="combination4"):
        super(FirstVariantOfConvNet, self).__init__()
        print('FirstVariantOfConvNet Loaded')

        affinity_graph_dims = [ag_init_dim, 512, 256]

        drug_transform_dims = [affinity_graph_dims[-1], 1024, embedding_dim]
        target_transform_dims = [affinity_graph_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, affinity_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):

        num_node1s, num_node2s = affinity_graph.num_node1s, affinity_graph.num_node2s

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1]

        drug_output_embedding = drug_transform_embedding
        target_output_embedding = target_transform_embedding

        return drug_output_embedding, target_output_embedding


class SecondVariantOfConvNet(torch.nn.Module):
    def __init__(self, ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, affinity_dropout_rate=0.2, skip=False, embedding_dim=128, integration_mode="combination4"):
        super(SecondVariantOfConvNet, self).__init__()
        print('SecondVariantOfConvNet')

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[-1]
        target_output_embedding = self.target_output_linear(target_graph_embedding)[-1]

        return drug_output_embedding, target_output_embedding
    
    
class ThirdVariantOfConvNet(torch.nn.Module):
    def __init__(self, ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, affinity_dropout_rate=0.2, skip=False, embedding_dim=128, integration_mode="combination4"):
        super(ThirdVariantOfConvNet, self).__init__()
        print('ThirdVariantOfConvNet Loaded')

        affinity_graph_dims = [ag_init_dim, 512, 256]

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4]

        drug_transform_dims = [affinity_graph_dims[-1], 1024, embedding_dim]
        target_transform_dims = [affinity_graph_dims[-1], 1024, embedding_dim]

        drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
        target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]

        self.integration_func, integration_dim_func = vector_operations[integration_mode]
        self.output_dim = integration_dim_func(embedding_dim)

        self.affinity_graph_conv = DenseGCNModel(affinity_graph_dims, affinity_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

        self.drug_graph_conv = GCNModel(drug_graph_dims)
        self.target_graph_conv = GCNModel(target_graph_dims)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])

    def forward(self, affinity_graph, drug_graph_batchs, target_graph_batchs, drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):

        num_node1s, num_node2s = affinity_graph.num_node1s, affinity_graph.num_node2s

        affinity_graph_embedding = self.affinity_graph_conv(affinity_graph)[-1]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(affinity_graph_embedding[:num_node1s])[-1]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(affinity_graph_embedding[num_node1s:])[-1]

        drug_graph_embedding = self.drug_graph_conv(drug_graph_batchs)[-1]
        target_graph_embedding = self.target_graph_conv(target_graph_batchs)[-1]

        drug_output_embedding = self.integration_func(self.drug_output_linear(drug_graph_embedding)[-1], drug_transform_embedding)
        target_output_embedding = self.integration_func(self.target_output_linear(target_graph_embedding)[-1], target_transform_embedding)

        return drug_output_embedding, target_output_embedding


class Predictor(torch.nn.Module):
    def __init__(self, embedding_dim=128, output_dim=1, prediction_mode="cat"):
        super(Predictor, self).__init__()
        print('Predictor Loaded')

        self.prediction_func, prediction_dim_func = vector_operations[prediction_mode]
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding):

        drug_id, target_id, y = data.drug_id, data.target_id, data.y

        drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
        target_feature = target_embedding[target_id.int().cpu().numpy()]

        concat_feature = self.prediction_func(drug_feature, target_feature)

        mlp_embeddings = self.mlp(concat_feature)
        link_embeddings = mlp_embeddings[-2]
        out = mlp_embeddings[-1]

        return out, link_embeddings
