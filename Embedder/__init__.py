import torch
import torch.nn as nn

from Embedder.positional_encoding import Embedder
from Embedder.hash_encoding import HashEmbedder 
from Embedder.spherical_harmonic import SHEncoder

def get_embedders(name, args, multires=None, bbox=None):
    """
    none: no embedding
    pos: Standard positional encoding (Nerf, section 5.1)
    hash: Hashed pos encoding
    sh: Spherical harmonic encoding
    """
    if name == "none":
        return nn.Identity(), 3
    elif name == "pos":
        assert multires is not None
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires - 1,
                    'num_freqs' : multires, 
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif name == "hash":
        assert bbox is not None
        embed = HashEmbedder(bounding_box=bbox, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
    elif name == "sh":
        embed = SHEncoder()
        out_dim = embed.out_dim
    else:
        raise ValueError(f"Invalid embedding type {name}")

    return embed, out_dim

def torch_embedding(samples, hdim):
    """
    For appearance or transient embedding
    """
    return nn.Embedding(samples, hdim) 