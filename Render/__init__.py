import argparse 
import torch

from collections import defaultdict 
from einops import rearrange

from .sampling.coarse import sample
from .sampling.fine import sample_pdf
from .ray_parser import parse_rays, post_process
from .embed import embed_all
from .interpret.nerf import raw2outputs
from ray_util import *
from load.load_data import CameraConfig

class BatchRender:
    def __init__(self, cc: CameraConfig,
                 models: dict, embedders: dict,
                 usage: dict, args: argparse.Namespace,
                 bbox):
        """
        NeRF volumetric rendering

        models:
            coarse and fine models for predicting RGB and density

        embedders:
            embedders for positional ennc. & viewdir ennc.

        Special case (testing):
            perturb = False 
            raw_noise_std = 0

        seed: random seed 
        dataset_type: dataset type (equirect, blender, etc)

        render_bsz: num of rays processed/rendered in parallel, adjust to prevent OOM
        net_bsz: num of rays sent to model in parallel (render_bsz % net_bsz == 0)
        perturb: 0 for uniform sampling, 1 for jittered (stratified rand points) sampling
        N_coarse: num of coarse samples per ray
        N_fine: num of additional fine samples per ray
        white_bkgd: whether to assume white background
        raw_noise_std: std dev of noise added to regularize sigma_a output, 1e0 recommended
        lindisp: If True, sample linearly in inverse depth rather than in depth.
        """     
        self.cc = cc
        self.models = models
        self.embedders = embedders
        self.usage = usage
        self.bbox = bbox
        self.args = args

        self.perturb_test = False 
        self.raw_noise_std_test = 0

        if self.seed is not None:
            torch.manual_seed(self.args.seed)

    def __call__(self, rays, verbose=False, 
                 retraw=False, test_mode=False):
        render_bsz = rays.shape[0]
        if test_mode is True:
            perturb = self.perturb_test
            raw_noise_std = self.raw_noise_std_test
        else:
            perturb = self.args.perturb
            raw_noise_std = self.args.raw_noise_std

        near = rays[:, 6].unsqueeze(-1) # (render_bsz, 1)
        far = rays[:, 7].unsqueeze(-1) # (render_bsz, 1)
        coarse_samples = sample(N_coarse=self.args.N_coarse, 
                         near=near, far=far, 
                         lindisp=self.args.lindisp, 
                         perturb=perturb)
        
        inputs = parse_rays(rays=rays, samples=coarse_samples, usage=self.usage)
        embedded = embed_all(inputs=inputs, embedders=self.embedders)

        coarse_raw = []
        for i in range(0, embedded.shape[0], self.net_bsz):
            coarse_raw.append(self.models["coarse"](embedded[i:i + self.net_bsz]))
        coarse_raw = post_process(coarse_raw)
        # (render_bsz, N_coarse, ?)
        coarse_raw = rearrange(coarse_raw, '(r cs) c -> r cs c', 
                                r=render_bsz, cs=self.args.N_coarse)

        coarse_out = raw2outputs(raw=coarse_raw, samples=coarse_samples, 
                                rays_d=rays_d, white_bkgd=self.args.white_bkgd,
                                raw_noise_std=raw_noise_std)

        if self.N_fine > 0:
            weights = coarse_out.weights
            fine_samples = sample_pdf(N_fine=self.args.N_fine,
                                      z_vals=coarse_samples, 
                                      weights=weights[:,1:-1],
                                      perturb=perturb)
            inputs = parse_rays(rays=rays, samples=fine_samples, usage=self.usage)
            embedded = embed_all(inputs=inputs, embedders=self.embedders)   

            model_name = "fine" if self.models["fine"] is not None else "coarse"
            fine_raw = []
            for i in range(0, embedded.shape[0], self.net_bsz):
                fine_raw.append(self.models[model_name](embedded[i:i + self.net_bsz]))
            fine_raw = post_process(fine_raw)
            # (render_bsz, N_fine, ?)
            fine_raw = rearrange(fine_raw, '(r cs) c -> r cs c', 
                                r=render_bsz, cs=self.args.N_fine)

            fine_out = raw2outputs(raw=fine_raw, samples=fine_samples, 
                                    rays_d=rays_d, white_bkgd=self.args.white_bkgd,
                                    raw_noise_std=raw_noise_std)

        ret = {}
        for k in coarse_out:
            ret[f"coarse_{k}"] = coarse_out[k]
        if self.N_fine > 0:
            for k in fine_out:
                ret[f"fine_{k}"] = fine_out[k]
        return ret
    
class Render:
    def __init__(self, bsz: int, batch_render: BatchRender):
        self.bsz = bsz
        self.batch_render = batch_render

    def __call__(self, rays, **kwargs):
        """
        Render in batches of size bsz
        """
        total_rays = rays.shape[0]

        outputs = defaultdict(list)
        for i in range(0, total_rays, self.bsz):
            # (render_bsz, ?)
            batch_output = self.batch_render(rays[i:i + self.bsz], **kwargs)
            for k, v in batch_output.items():
                outputs[k] += [v]           

        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0) 

        return outputs

