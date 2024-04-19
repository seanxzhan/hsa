import torch
from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn import global_mean_pool


class OBBGNN(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 graph_feature_dim,
                 output_dims,
                 decoder_hidden=3,
                 decoder_internal_dims=32):
        super().__init__()

        self.gcn = GCN(num_node_features, output_dims=graph_feature_dim)
        self.attention = Attention(graph_feature_dim)
        
        net = (
            torch.nn.Linear(graph_feature_dim, decoder_internal_dims, bias=False),
            torch.nn.ReLU())
        for _ in range(decoder_hidden-1):
            net = net + (
                torch.nn.Linear(decoder_internal_dims, decoder_internal_dims, bias=False),
                torch.nn.ReLU())
        net = net + (torch.nn.Linear(decoder_internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, node_feat, adj, mask, batch):
        # obbs_att = self.attention(obbs)
        obbs_feat = self.gcn(node_feat, adj, mask, batch)
        obbs_feat = self.attention(obbs_feat)
        out = self.net(obbs_feat)
        return out
    

class GCN(torch.nn.Module):
    def __init__(self,
                 num_node_features,
                 internal_dims=128,
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
        # x = global_mean_pool(x.view(-1, self.output_dims), batch)  # [batch_size, hidden_channels]

        return x


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