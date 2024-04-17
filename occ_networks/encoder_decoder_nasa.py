import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.data import Data

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = PointNetEnc()
        self.dec = SmallMLPs()

    def forward(self, surface_points: Data, batch, query_points):
        z, mu, log_var = self.enc(None, surface_points, batch)
        z = z.unsqueeze(1).expand(-1, query_points.shape[1], -1)
        occs = self.dec(query_points, z)
        return occs, mu, log_var


class SDFVae(torch.nn.Module):
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
        
    def forward(self, points, features, mask=None):
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


class PointNetEnc(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 32, 32, 64]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([64 + 3, 64, 64, 128]))
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 128, 256, 512]))

        # MLP for generating mean and log variance of the latent Gaussian
        self.fc_mu = nn.Linear(512, 32)  # Latent space size can be adjusted
        self.fc_log_var = nn.Linear(512, 32)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    

class SmallMLPs(torch.nn.Module):
    def __init__(self,
                 input_dims=3,
                 feature_dims=32,
                 internal_dims=64,
                 hidden=4,
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
        p = torch.concatenate((p, feature), dim=-1)
        out = self.net(p)
        return out
    

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


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
