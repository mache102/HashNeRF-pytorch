import torch 

from einops import rearrange, reduce 


def sample_pdf(N_fine, z_vals, weights, perturb=False, eps=1e-5):
    """
    Sample a pdf

    Args:
        z_vals: (render_bsz, N_coarse - 1)
        weights: (render_bsz, N_coarse - 2)
        perturb: bool, False for uniform sampling, 
                True for jittered (stratified rand points) sampling
        eps: int, small value to prevent nans

    create a cdf from the weights obtained from the coarse model
    Returns:
        samples: (render_bsz, N_fine)
    """
    # mid
    z_vals_mid = .5 * (z_vals[:,1:] + z_vals[:,:-1]) # (render_bsz, N_coarse - 1)

    # Get pdf
    weights = weights + eps # prevent nans
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum')
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (render_bsz, N_coarse-1)

    # Take uniform samples
    if perturb:
        u = torch.rand((cdf.shape[0], N_fine))
    else:
        u = torch.linspace(0, 1, N_fine)
        u = u.expand((cdf.shape[0], N_fine))

    # Invert CDF
    u = u.contiguous()
    idxs = torch.searchsorted(cdf, u, right=True) # (render_bsz, N_fine)
    below = torch.clamp_min(idxs - 1, 0)
    above = torch.clamp_max(idxs, cdf.shape[-1] - 1)
    idxs_g = torch.stack([below, above], -1)  # (render_bsz, N_fine, 2)

    # cdf_g = tf.gather(cdf, idxs_g, axis=-1, batch_dims=len(idxs_g.shape)-2)
    # bins_g = tf.gather(bins, idxs_g, axis=-1, batch_dims=len(idxs_g.shape)-2)
    matched_shape = [idxs_g.shape[0], idxs_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, idxs_g)
    bins_g = torch.gather(z_vals.unsqueeze(1).expand(matched_shape), 2, idxs_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

    # # (render_bsz, N_fine * 2)
    # idxs_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2) 
    # cdf_g = rearrange(torch.gather(cdf, 1, idxs_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    # z_vals_g = rearrange(torch.gather(z_vals_mid, 1, idxs_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    # denom = cdf_g[...,1] - cdf_g[...,0] # (render_bsz, N_fine)
    # denom[denom < eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    #                     # anyway, therefore any value for it is fine (set to 1 here)
    # samples = z_vals_g[...,0] + (u - cdf_g[...,0]) / denom * (z_vals_g[...,1] - z_vals_g[...,0])
    # samples = samples.detach()

    # samples, _ = torch.sort(torch.cat([z_vals, samples], -1), -1)
    
    # return samples # (render_bsz, N_fine)
