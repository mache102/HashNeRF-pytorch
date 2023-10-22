import torch.nn as nn

from Embedder.positional_encoding import PosEmbedder
from Embedder.hash_encoding import HashEmbedder 
from Embedder.spherical_harmonic import SHEncoder

def get_embedders(embedder_config, dataset):
    embedder_names = ["xyz", "dir", "appearance", "transient"]
    embedders = {}

    for k in embedder_names:
        if embedder_config.get(k) is None:
            embedders[k] = None
            print(f"{k.upper()} embedder: None")
            continue

        embedders[k] = get_embedder(embedder_config[k], bbox=dataset.bbox)
        print(f"{k.upper()} embedder: {embedder_config[k]['name']}")

    return embedders

def get_embedder(config, bbox=None):
    """
    none: no embedding
    pos: Standard positional encoding (Nerf, section 5.1)
    hash: Hashed pos encoding
    sh: Spherical harmonic encoding
    """
    name = config["type"]
    config = config["config"]

    if name == "none":
        return nn.Identity(), 3
    elif name == "positional":
        embedder = PosEmbedder(max_log2=config["max_log2"], N_freqs=config["N_freqs"], \
                            log_sampling=config["log_sampling"], \
                            include_input=config["include_input"])
    elif name == "hash":
        assert bbox is not None
        embedder = HashEmbedder(n_levels=config["n_levels"], \
                            n_features_per_level=config["n_features_per_level"], \
                            log2_hashmap_size=config["log2_hashmap_size"], \
                            base_resolution=config["base_resolution"], \
                            finest_resolution=config["finest_resolution"], \
                            bounding_box=bbox)
    elif name == "sh":
        embedder = SHEncoder()
    elif name == "torch":
        embedder = torch_embedding(config["samples"], config["hdim"])
    else:
        raise ValueError(f"Invalid embedding type {name}")

    return embedder

def torch_embedding(samples, hdim):
    """
    For appearance or transient embedding
    """
    return nn.Embedding(samples, hdim) 