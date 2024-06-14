import torch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn import global_mean_pool


class OBBGNN(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 graph_feature_dim,
                 num_nodes=4):
        super().__init__()

        self.gcn = GCN(num_node_features, output_dims=graph_feature_dim)
        # self.attention = Attention(graph_feature_dim)
        self.decoder = GraphDecoder(input_dims=graph_feature_dim, num_nodes=num_nodes)
                

    def forward(self, node_feat, adj, mask, batch):
        # obbs_att = self.attention(obbs)
        z = self.gcn(node_feat, adj, mask, batch)
        node_feat, edge_feat = self.decoder(z)
        # obbs_feat = self.attention(obbs_feat)
        # out = self.net(obbs_feat)
        # return out
        return node_feat, edge_feat
    

class GCN(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 internal_dims=64,
                 hidden=5,
                 output_dims=64):
        super().__init__()
        
        self.output_dims = output_dims

        net = (
            DenseGCNConv(num_node_features, internal_dims),
            torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (
                DenseGCNConv(internal_dims, internal_dims),
                torch.nn.ReLU())
        net = net + (DenseGCNConv(internal_dims, output_dims),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, x, adj, mask, batch):
        relu = 0
        for layer in self.net:
            if relu:
                x = layer(x)
                relu = 0
            else:
                x = layer(x, adj, mask)
                relu = 1
        # 2. Readout layer
        # print(x.shape)
        # print(mask)
        x = global_mean_pool(x.view(-1, self.output_dims), batch)  # [batch_size, hidden_channels]

        return x


class GraphDecoder(torch.nn.Module):
    def __init__(self, input_dims, num_nodes,
                 node_feature_dims=3, edge_feature_dims=3,
                 node_feature_internal_dims=64,
                 edge_feature_internal_dims=64,
                 node_feature_hidden=4,
                 edge_feature_hidden=4):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feature_dims = node_feature_dims
        self.edge_feature_dims = edge_feature_dims

        node_net = (
            torch.nn.Linear(input_dims, node_feature_internal_dims, bias=False),
            torch.nn.ReLU())
        for _ in range(node_feature_hidden-1):
            node_net = node_net + (
                torch.nn.Linear(node_feature_internal_dims, node_feature_internal_dims, bias=False),
                torch.nn.ReLU())
        node_net = node_net + (torch.nn.Linear(node_feature_internal_dims, num_nodes * node_feature_dims, bias=False),)
        self.node_net = torch.nn.Sequential(*node_net)

        edge_net = (
            torch.nn.Linear(input_dims, edge_feature_internal_dims, bias=False),
            torch.nn.ReLU())
        for _ in range(edge_feature_hidden-1):
            edge_net = edge_net + (
                torch.nn.Linear(edge_feature_internal_dims, edge_feature_internal_dims, bias=False),
                torch.nn.ReLU())
        # edge_net = edge_net + (torch.nn.Linear(edge_feature_internal_dims, num_nodes * num_nodes * edge_feature_dims, bias=False),)
        edge_net = edge_net + (torch.nn.Linear(edge_feature_internal_dims, num_nodes * edge_feature_dims, bias=False),)
        self.edge_net = torch.nn.Sequential(*edge_net)

    def forward(self, z):
        node_feats = self.node_net(z).view(-1, self.num_nodes, self.node_feature_dims)
        # edge_feats = self.edge_net(z).view(-1, self.num_nodes, self.num_nodes, self.node_feature_dims)
        edge_feats = self.edge_net(z).view(-1, self.num_nodes, self.node_feature_dims)
        return node_feats, edge_feats
        # return None, edge_feats


class Attention(torch.nn.Module):

    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        self.key_dim = 32
        self.query_w = torch.nn.Linear(hidden_dim, self.key_dim)
        self.key_w = torch.nn.Linear(hidden_dim, self.key_dim)
        self.value_w = torch.nn.Linear(hidden_dim, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        # self.scale = 1 / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))

    def forward(self, x):
        queries = self.query_w(x)
        keys = self.key_w(x)
        vals = self.value_w(x)
        # attention = self.softmax(torch.einsum('bgqf,bgkf->bgqk', queries, keys))  
        # out = torch.einsum('bgvf,bgqv->bgqf', vals, attention)
        keys = keys.transpose(1, 2)
        attention = self.softmax(torch.einsum('bij,bjk->bik', queries, keys))  
        out = torch.einsum('bij,bjk->bik', attention, vals)
        out = self.gamma * out + x
        return out