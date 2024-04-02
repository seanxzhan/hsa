import math
import torch

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
        p = torch.concatenate((p, feature), dim=-1)
        out = self.net(p)
        return out


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

        for i in range(21):
            self.nets.append(SmallMLPs(input_dims,
                                       feature_dims,
                                       internal_dims,
                                       hidden,
                                       output_dims,
                                       multires))
        
        # for i, mlp in enumerate(self.nets):
        #     if i not in [6, 13, 18]:
        #         for param in mlp.parameters():
        #             param.requires_grad = False

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
