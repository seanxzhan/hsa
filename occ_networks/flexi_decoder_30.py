import math
import torch
# from align_networks.gnn_dense_ae_58 import OBBGNN
from tqdm import tqdm


class SDFDecoder(torch.nn.Module):
    """Given a query point, output the SDF
    """
    def __init__(self,
                 input_dims=3,
                 num_parts=4,
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
        # self.nets = torch.nn.ModuleList()
        # for i in range(num_parts):
        #     self.nets.append(SmallMLPs(input_dims,
        #                                feature_dims,
        #                                internal_dims,
        #                                hidden,
        #                                output_dims,
        #                                multires))

        self.vol_cell_dim = 32

        refine_out_dims = 4
        self.refine_net = SmallMLPs(input_dims,
                                    self.vol_cell_dim,
                                    internal_dims,
                                    hidden,
                                    refine_out_dims,
                                    multires)
        
        # self.feature_volume = VoxDecoder(feature_size=feature_dims)

    def get_sdf_deform(self, points):
        return self.refine_net.forward(points)
        # print(features.shape)
        # return self.refine_net.forward(points, features.expand(points.shape[0], -1))
    
    def pre_train_sphere(self, iter):
        print("Initialize SDF to sphere")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-4)

        for i in tqdm(range(iter)):
            p = torch.rand((1024,3), device='cuda') - 0.5
            ref_value  = torch.sqrt((p**2).sum(-1)) - 0.3
            # output = self.get_sdf_deform(p, torch.zeros((1, 128)).to('cuda'))
            output = self.get_sdf_deform(p)
            loss = loss_fn(torch.tanh(output[...,0]), ref_value)
            # loss = loss_fn(output[...,0], ref_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Pre-trained MLP", loss.item())

    # def get_occ(self, points, features, mask=None):
    #     # xforms: batch size x 4 x 3
    #     batch_size, _, n_points, _ = points.shape

    #     occs = torch.zeros((batch_size,
    #                         n_points,
    #                         self.num_parts)).to(points.device)
    #     for i in range(self.num_parts):
    #         pts = points[:, i, :, :]
    #         feat = features[:, i*self.feature_dims:(i+1)*self.feature_dims]
    #         feat = feat.unsqueeze(1).expand(-1, pts.shape[1], -1)
    #         occs[:, :, i] = self.nets[i](pts, feat).squeeze(-1)

    #     out = occs
    #     return out


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

from torch.nn import Linear, ConvTranspose3d, ReLU, Sigmoid, Sequential

class VoxDecoder(torch.nn.Module):
    def __init__(self,
                 feature_size = 64) -> None:
        super().__init__()

        # net = (Linear(feature_size, 256), ReLU())
        self.fc = Sequential(Linear(feature_size, 128), ReLU())
        net = (ConvTranspose3d(in_channels=128,
                                out_channels=128,
                                kernel_size=4,
                                stride=3,
                                padding=0),
                ReLU())
        net += (ConvTranspose3d(in_channels=128,
                                out_channels=64,
                                kernel_size=4,
                                stride=2,
                                padding=1),
                ReLU())
        net += (ConvTranspose3d(in_channels=64,
                                out_channels=32,
                                kernel_size=4,
                                stride=2,
                                padding=1),
                ReLU())
        net += (ConvTranspose3d(in_channels=32,
                                out_channels=32,
                                kernel_size=4,
                                stride=2,
                                padding=1),)
        # net += (ConvTranspose3d(in_channels=8,
        #                         out_channels=8,
        #                         kernel_size=1,
        #                         stride=1,
        #                         padding=0),
        #         )
        # net += (Sigmoid(),)

        self.net = Sequential(*net)

    def forward(self, feature):
        mapped = self.fc(feature)
        x = mapped.view(-1, 128, 1, 1, 1)
        x = self.net(x)
        # print(x.shape)
        return x
        # for layer in self.net:
        #     x = layer(x)
        #     print(x.shape)
        # print("done")
        # return x
        