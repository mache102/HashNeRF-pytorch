import torch 
import pdb 
from einops import rearrange 
from torch.distributions import Categorical

from math_util import alpha_composite 


def raw2outputs(raw: torch.Tensor, samples: torch.Tensor,
                rays_d: torch.Tensor, usage: dict,
                white_bkgd: bool = False, raw_noise_std: float = 0.0):
    """
    Transforms model's predictions to semantically meaningful values.
    """
    results = {}

    # differences of z vals = dists. last value is 1e10 (astronomically large in the context)
    dists = samples[:, 1:] - samples[:, :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[:,:1].shape)], -1)
    dists = dists * torch.norm(rearrange(rays_d, 'r c -> r () c'), dim=-1) # (render_bsz, samples)

    static_sigmas = raw[...,3]    
    if raw_noise_std > 0:
        static_sigmas += torch.randn(static_sigmas.shape) * raw_noise_std

    static_alphas = alpha_composite(static_sigmas, dists) # (render_bsz, N_coarse)
    if usage["transient"]:
        transient_rgbs = raw[..., 4:7]
        transient_sigmas = raw[..., 7]
        transient_betas = raw[..., 8]

        transient_alphas = alpha_composite(transient_sigmas, dists) 
        alphas = alpha_composite(static_sigmas + transient_sigmas, dists) 

    ones = torch.ones((static_alphas.shape[0], 1)) # for the prod start with 1s
    weights = static_alphas * torch.cumprod(torch.cat([ones, 1. - static_alphas + 1e-10], -1), -1)[:, :-1] 

    static_rgbs = torch.sigmoid(raw[...,:3]) # (render_bsz, samples, 3)
    static_rgbs = torch.sum(weights[...,None] * static_rgbs, -2)  # (render_bsz, 3)  
    accumulation = torch.sum(weights, -1)
    if white_bkgd:
        static_rgbs += (1. - accumulation[...,None])
    results["static_rgbs"] = static_rgbs    
    results["accumulation"] = accumulation

    depth = torch.sum(weights * samples, -1) / torch.sum(weights, -1)
    disparity = 1./torch.max(1e-10 * torch.ones_like(depth), depth)
    results["depth"] = depth
    results["disparity"] = disparity

    results['transient_sigmas'] = transient_sigmas

    # Calculate weights sparsity loss
    try:
        entropy = Categorical(probs = torch.cat([weights, 1.0 - weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
    except:
        print("Something occured in calculating weights sparsity loss. Begin debugging...")
        pdb.set_trace()
    results["weights"] = weights
    results["sparsity_loss"] = entropy 

    return results 