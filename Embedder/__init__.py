import torch.nn as nn

from .positional_encoding import PosEmbedder
from .hash_encoding import HashEmbedder 
from .spherical_harmonic import SHEncoder

def get_embedders(embedder_config, dataset):
    embedder_names = ["xyz", "dir", "appearance", "transient"]
    embedders = {}

    # note that some embedders may be trainable, i.e. the linear embeddings for appearance and transient
    is_trainable = {} 

    for k in embedder_names:
        if embedder_config.get(k) is None:
            embedders[k] = None
            is_trainable[k] = False
            print(f"{k.upper()} embedder: None")
            continue

        embedders[k], is_trainable[k] = get_embedder(embedder_config[k], bbox=dataset.bbox)
        print(f"{k.upper()} embedder: {embedder_config[k]['name']}")
        if is_trainable[k]: 
            print("Embedder is trainable")

    return embedders, is_trainable

def get_embedder(config, bbox=None):
    """
    none: no embedding
    pos: Standard positional encoding (Nerf, section 5.1)
    hash: Hashed pos encoding
    sh: Spherical harmonic encoding
    """
    name = config["type"]
    config = config["config"]

    is_trainable = False

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
    elif name == "linear": # linear embedding (is there a better name?)
        embedder = TorchEmbedding(config["samples"], config["hdim"]) 
        is_trainable = True 
    else:
        raise ValueError(f"Invalid embedding type {name}")

    return embedder, is_trainable

class TorchEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(TorchEmbedding, self).__init__(num_embeddings, embedding_dim)

    @property
    def out_dim(self):
        """name consistency with other embedding classes"""
        return self.embedding_dim

