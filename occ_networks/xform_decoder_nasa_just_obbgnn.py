import math
import torch
from align_networks.gnn_dense_att import OBBGNN



class SDFDecoder(torch.nn.Module):
    """Given a query point, output the SDF
    """
    def __init__(self,
                 input_dims=3,
                 num_parts=42,
                 feature_dims=8,
                 internal_dims=32,
                 hidden=3,
                 obb_decoder_internal_dims=32,
                 obb_decoder_hidden=3,
                 output_dims=1,
                 multires=1) -> None:
        super().__init__()
        self.embed_fn = None

        self.num_parts = num_parts
        self.feature_dims = feature_dims
        # self.nets = torch.nn.ModuleList()
        
        self.obb_gnn = OBBGNN(num_node_features=3,
                              graph_feature_dim=32,
                              output_dims=3,
                              decoder_hidden=obb_decoder_hidden,
                              decoder_internal_dims=obb_decoder_internal_dims)
        
    def learn_xform(self, node_feat, adj, mask, batch):
        return self.obb_gnn.forward(node_feat, adj, mask, batch)

