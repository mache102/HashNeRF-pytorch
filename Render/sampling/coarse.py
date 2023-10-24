import torch

def sample(N_coarse, near, far, lindisp=False, perturb=False):
    """
    Sample 

    Args:
        near: (render_bsz, 1)
        far: (render_bsz, 1)
        lindisp: bool, inverse linear sampling
        perturb: bool, False for uniform sampling, 
                True for jittered (stratified rand points) sampling

    Returns: 
        samples: (render_bsz, N_coarse)
    """
    # prepare for sampling
    t_vals = torch.linspace(0, 1, N_coarse)
    # (render_bsz, N_coarse)
    if lindisp:
        samples = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    else:
        samples = near * (1. - t_vals) + far * (t_vals)

    if perturb:
        # get intervals between samples
        # mid: (render_bsz, N_coarse - 1)
        # upper, lower: same as samples
        mids = .5 * (samples[:, 1:] + samples[:, :-1])
        upper = torch.cat([mids, samples[:, -1:]], -1)
        lower = torch.cat([samples[:, :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(samples.shape)
        samples = lower + (upper - lower) * t_rand # same shape

    return samples