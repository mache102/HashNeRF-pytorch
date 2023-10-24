"""
Multiresolution hash encoding as described in instant-ngp paper
"""
import torch
import torch.nn as nn

HASH_PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

# smth weird happened with itertools.product so it's just hardcoded
BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                            device='cuda')

class HashEmbedder(nn.Module):
    def __init__(self, bounding_box, n_levels=16, n_features_per_level=2,\
                log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        """
        Hash embedder as described in instant-ngp paper
        
        bounding_box: (2, 3), min and max x,y,z coordinates of object bbox
        n_levels: int, number of embedding levels
        n_features_per_level: int, number of features per level
        log2_hashmap_size: int, log2 of hashmap size
        ^ Pareto optimum in paper is 16 (2^16 = 65536)

        base_resolution: int, number of voxels per axis at the coarsest level
        finest_resolution: int, number of voxels per axis at the finest level

        out_dim: int, total number of features
        
        b: float, geometric progression factor
        ^ a resolution at level i can be obtained as 
        resolution = torch.floor(self.base_resolution * self.b**i)
        at i=0, the resolution is base_resolution
        at i=n_levels-1, the resolution is finest_resolution

        embeddings: list of nn.Embedding objects
        ^ each embedding is of size 2**log2_hashmap_size x n_features_per_level
        and initialized with uniform random values in [-0.0001, 0.0001]

        """
        super(HashEmbedder, self).__init__()
        self.bounding_box = bounding_box
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = torch.tensor(base_resolution)
        self.finest_resolution = torch.tensor(finest_resolution)
        self.out_dim = self.n_levels * self.n_features_per_level

        self.b = torch.exp((torch.log(self.finest_resolution)-torch.log(self.base_resolution))/(n_levels-1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size, \
                                        self.n_features_per_level) for i in range(n_levels)])
        # custom uniform initialization
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)
            # self.embeddings[i].weight.data.zero_()

    def get_voxel_vertices(self, resolution: torch.Tensor):
        """
        resolution: number of voxels per axis
        """
        box_min, box_max = self.bounding_box

        if not torch.all(self.xyz <= box_max) or not torch.all(self.xyz >= box_min):
            # print("ALERT: some points are outside bounding box. Clipping them!")
            self.xyz = torch.clamp(self.xyz, min=box_min, max=box_max)


        grid_size = (box_max - box_min)/resolution
        
        bottom_left_idx = torch.floor((self.xyz - box_min) / grid_size).int()
        voxel_min_vertex = bottom_left_idx * grid_size + box_min
        voxel_max_vertex = voxel_min_vertex + torch.tensor([1.0,1.0,1.0])*grid_size


        voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
        hashed_voxel_indices = hash(voxel_indices, self.log2_hashmap_size)

        return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

    def forward(self, x):
        """
        Embed an input point cloud x
        
        x: (net_bsz, 3), 3D coordinates of samples
        ^ we define a temporary class attr self.xyz to store x
        for use in get_voxel_vertices

        For each level, we compute the voxel vertices and the corresponding
        voxel embeddings. Then we interpolate the embeddings at the input
        coordinates x. 
        """
        # set x as a class attr 
        self.xyz = x

        x_embedded = []
        for i in range(self.n_levels):
            resolution = torch.floor(self.base_resolution * self.b**i)
            voxel_min_vertex, voxel_max_vertex, \
                hashed_voxel_indices, keep_mask = \
                self.get_voxel_vertices(resolution=resolution)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices)
            x_embedded.append(trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds))

        # clip if some points are outside bounding box
        keep_mask = self.xyz == torch.max(torch.min(self.xyz, box_max), box_min)
        keep_mask = keep_mask.sum(dim=-1)==keep_mask.shape[-1]
        
        return torch.cat(x_embedded, dim=-1), keep_mask

def hash(coords, log2_hashmap_size):
    """
    Spatial hash function, Teschner et al. 2003
    h(x) = ( bitwise_xor_{i=1}^{d}x_i * \pi_i ) mod T

    This function can process upto 7 dim coordinates

    coords: (net_bsz, 8, dim)
    """

    xor_result = torch.zeros_like(coords)[..., 0]
    device = xor_result

    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * HASH_PRIMES[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(device) & xor_result

def trilinear_interp(x, min_vertex, max_vertex, embedds):
    """
    Trilinear interpolation for a batch of 3d points

    Source: https://en.wikipedia.org/wiki/Trilinear_interpolation

    x: (net_bsz, 3)
    min_vertex: (net_bsz, 3)
    max_vertex: (net_bsz, 3)
    embedds: (net_bsz, 8, 2)
    """

    # normalize to vertex coordinates
    weights = (x - min_vertex) / (max_vertex - min_vertex) # (net_bsz, 3)

    # Step 1
    # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
    # 3d -> 2d
    # [:, None] notation is equivalent to [:, np.newaxis] or .reshape(-1, 1)
    c00 = embedds[:,0] * (1 - weights[:,0][:,None]) + embedds[:,4] * weights[:,0][:,None]
    c01 = embedds[:,1] * (1 - weights[:,0][:,None]) + embedds[:,5] * weights[:,0][:,None]
    c10 = embedds[:,2] * (1 - weights[:,0][:,None]) + embedds[:,6] * weights[:,0][:,None]
    c11 = embedds[:,3] * (1 - weights[:,0][:,None]) + embedds[:,7] * weights[:,0][:,None]

    # Step 2
    # 2d -> 1d
    c0 = c00 * (1 - weights[:,1][:,None]) + c10 * weights[:,1][:,None]
    c1 = c01 * (1 - weights[:,1][:,None]) + c11 * weights[:,1][:,None]

    # Step 3
    # interpolate along z-axis
    c = c0 * (1 - weights[:,2][:,None]) + c1 * weights[:,2][:,None]

    return c
