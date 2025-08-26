from tqdm import tqdm
import torch
# from align_networks.gnn_dense_ae_58 import OBBGNN
from align_networks.occflexi_gnn_18 import OBBGNN


class GNN(torch.nn.Module):
    """Given a query point, output the SDF
    """
    def __init__(self,
                 num_parts=42,
                 feature_dims=8) -> None:
        super().__init__()
        self.embed_fn = None
        self.num_parts = num_parts
        self.feature_dims = feature_dims
        
        self.obb_gnn = OBBGNN(num_node_features=32,
                              graph_feature_dim=32,
                              num_parts=num_parts)
        
    def learn_geom_xform(self, node_feat, adj, mask, batch):
        return self.obb_gnn.forward(node_feat, adj, mask, batch)
