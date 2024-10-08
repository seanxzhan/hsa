from tqdm import tqdm
import torch
# from align_networks.gnn_dense_ae_58 import OBBGNN
from align_networks.occflexi_gnn_15 import OBBGNN


class SDFDecoder(torch.nn.Module):
    """Given a query point, output the SDF
    """
    def __init__(self,
                 input_dims=3,
                 num_parts=42,
                 feature_dims=8,
                 internal_dims=32,
                 hidden=3,
                 output_dims=1,
                 multires=1) -> None:
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch

        self.num_parts = num_parts
        self.feature_dims = feature_dims
        self.nets = torch.nn.ModuleList()

        for i in range(num_parts):
            self.nets.append(SmallMLPs(input_dims,
                                       feature_dims,
                                       internal_dims,
                                       hidden,
                                       output_dims,
                                       multires))
        
        self.obb_gnn = OBBGNN(num_node_features=8,
                              graph_feature_dim=32,
                              num_parts=num_parts)
        
    def learn_geom_xform(self, node_feat, adj, mask, batch):
        return self.obb_gnn.forward(node_feat, adj, mask, batch)

    def forward(self, points, features, mask=None):
        # xforms: batch size x 4 x 3
        batch_size, _, n_points, _ = points.shape

        occs = torch.zeros((batch_size,
                            n_points,
                            self.num_parts)).to(points.device)
        for i in range(self.num_parts):
            pts = points[:, i, :, :]
            feat = features[:, i*self.feature_dims:(i+1)*self.feature_dims]
            feat = feat.unsqueeze(1).expand(-1, pts.shape[1], -1)
            occs[:, :, i] = self.nets[i](pts, feat).squeeze(-1)

        out = occs
        return out


class SmallMLPs(torch.nn.Module):
    def __init__(self,
                 input_dims=3,
                 feature_dims=8,
                 internal_dims=64,
                 hidden=3,
                 output_dims=1,
                 multires=1) -> None:
        super().__init__()
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            input_dims = input_ch
        
        net = (
            torch.nn.Linear(input_dims + feature_dims, internal_dims, bias=False),
            torch.nn.ReLU())
        for i in range(hidden-1):
            net = net + (
                torch.nn.Linear(internal_dims, internal_dims, bias=False),
                torch.nn.ReLU())
        net = net + (torch.nn.Linear(internal_dims, output_dims, bias=False),)
        self.net = torch.nn.Sequential(*net)

    def forward(self, p, feature):
        if self.embed_fn is not None:
            p = self.embed_fn(p)
        # p = torch.concatenate((p, feature), dim=-1)
        p = torch.concat((p, feature), dim=-1)
        out = self.net(p)
        return out
    
    def pre_train_sphere(self, iter):
        print("initializing SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for _ in tqdm(range(iter)):
            p = torch.rand((1, 1024,3), device='cuda') - 0.5
            distances = torch.sqrt((p**2).sum(-1))
            occupancy = (distances <= 0.3).float()
            output = self.forward(p, torch.zeros((1, 1024, 128)).to('cuda'))
            loss = loss_fn(output[...,0], occupancy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())


# Positional Encoding from https://github.com/yenchenlin/nerf-pytorch/blob/1f064835d2cca26e4df2d7d130daa39a8cee1795/run_nerf_helpers.py
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
