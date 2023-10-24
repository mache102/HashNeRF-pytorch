import argparse
import torch
import numpy as np

import imageio

from tqdm import trange, tqdm 
from Render import Render, BatchRender

from load.load_data import EquirectDataset
from Render.ray_parser import prepare_rays
from ray_util import * 
from util import *
from Loss import total_variation_loss

from math_util import to_8b, img2mse, mse2psnr

device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, dataset: EquirectDataset, models: dict,
                 optimizer: torch.optim.Optimizer, embedders: dict,
                 usage: dict, args: argparse.Namespace):

        self.args = args
        self.embedders = embedders
        self.models = models
        self.optimizer = optimizer
        self.usage = usage

        self.unpack_dataset(dataset)
        self.cc = dataset.cc

        self.savepath = args.savepath
        self.train_bsz = args.train_bsz
        self.global_step = args.global_step # curr step

        # start and end iters
        self.start = self.global_step + 1
        self.end = args.train_iters

        # starting index of the next training batch.
        # if this exceeds the number of training rays,
        # then we shuffle the dataset and reset this to 0
        self.next_batch_idx = 0 

        # constant
        self.lr = args.lr 
        self.lr_decay = args.lr_decay
        self.decay_rate = args.decay_rate

        # near and far bounds won't change during training
        self.near = torch.full((self.train_bsz, 1), self.cc.near)
        self.far = torch.full((self.train_bsz, 1), self.cc.far)

        batch_render = BatchRender(cc=self.cc, models=models,
                                    embedders=embedders, 
                                    usage=usage, args=args,
                                    bbox=dataset.bbox)
        self.render = Render(bsz=args.render_bsz, 
                             batch_render=batch_render)

    def unpack_dataset(self, dataset):
        """
        unpack train&test rays, bbox, and camera config
        """
        self.rays_train = shuffle_rays(all_to_tensor(dataset.rays_train, device))
        self.rays_test = all_to_tensor(dataset.rays_test, device)
        self.bbox = dataset.bbox

    def fit(self):
        """
        Training loop
        """
        print(f"Training iterations {self.start} to {self.end}")

        for iter in trange(self.start, self.end):
            # retrieve batch of data
            rays, targets = self.get_batch(start=self.next_batch_idx, 
                                            step=self.train_bsz)
            self.next_batch_idx += self.train_bsz

            # shuffle 
            if self.next_batch_idx >= self.rays_train.rgb.shape[0]:
                print("Shuffle data after an epoch!")
                self.rays_train = shuffle_rays(self.rays_train)
                self.next_batch_idx = 0 # reset

            # optimize
            # (train_bsz, ?) and (train_bsz,)
            og_shape = list(rays[:, 3].shape[:-1])

            outputs = self.render(rays=rays)
            outputs = {k: torch.reshape(v, og_shape + list(v.shape[1:])) for k, v in outputs.items()}
            loss, psnr = self.calc_loss(outputs, targets)
            self.update_lr()

            # Save state dicts for checkpoint
            if iter % self.args.iter_ckpt == 0:
                self.save_ckpt(iter)

            # Evaluate and save test set
            if iter % self.args.iter_test == 0 and iter > 0:
                if self.args.stage > 0:
                    test_fp = os.path.join(self.savepath, 'stage{}_test_{:06d}'.format(self.args.stage, iter))
                else:
                    test_fp = os.path.join(self.savepath, 'testset_{:06d}'.format(iter))

                self.eval_test(test_fp)

            if iter % self.args.iter_print == 0:
                tqdm.write(f"[TRAIN] Iter: {iter} Loss: {loss.item()}  PSNR: {psnr.item()}")

            self.global_step += 1

    def get_batch(self, start, step):
        """
        Retrieve a batch of rays (self.train_bsz)
        for training 
        use_batching is essentially True here
        """
        end = start + step   

        # (train_bsz, 3+3+1+1)
        rays = torch.cat([self.rays_train.o[start:end], 
                          self.rays_train.d[start:end],
                          self.near, self.far], -1)

        if self.usage["dir"]:
            # d is 3
            dir = rays[:, 3] / torch.norm(rays[:, 3], dim=-1, keepdim=True)
            rays = torch.cat([rays, dir[:, None]], -1)
   
        if self.usage["appearance"] or self.usage["transient"]: 
            rays = torch.cat([rays, self.rays_train.t[start:end]], -1)

        # all (train_bsz, 3)
        target = {
            "color": self.rays_train.rgb[start:end],
            "depth": self.rays_train.d[start:end], # d not depth?
        }
        return rays, target

    def calc_loss(self, outputs, targets):
        """
        Calculates loss. 
        supposts color loss (coarse+fine)
        and nerfw loss
        """
        self.optimizer.zero_grad()
        
        if self.usage["transient"]:
            loss_dict = {}
            loss_dict['c_l'] = 0.5 * ((outputs['coarse_color'] - targets['color'])**2).mean()
            if 'fine_color' in outputs:
                if 'beta' not in outputs: # no transient head, normal MSE loss
                    loss_dict['f_l'] = 0.5 * ((outputs['fine_color'] - targets['color'])**2).mean()
                else:
                    loss_dict['f_l'] = \
                        ((outputs['fine_color'] - targets['color'])**2/(2*outputs['beta'].unsqueeze(1)**2)).mean()
                    loss_dict['b_l'] = 3 + torch.log(outputs['beta']).mean() # +3 to make it positive
                    loss_dict['s_l'] = 0.01 * outputs['transient_sigma'].mean()

            loss = sum(l for l in loss_dict.values())
        else:
            # standard    
            prefix = 'fine' if self.args.N_fine > 0 else 'coarse'
            color_loss = img2mse(outputs[f'{prefix}_color'] - targets['color'])
            psnr = mse2psnr(color_loss)
            loss = color_loss

            if self.usage["depth"]:
                loss += torch.abs(outputs[f'{prefix}_depth'] - targets['depth']).mean()
            if self.usage["gradient"]:
                loss += img2mse(outputs[f'{prefix}_opacity'] - targets['opacity'])

            if self.args.N_fine > 0:
                prefix = "fine"
                loss += img2mse(outputs[f'{prefix}_color'] - targets['color'])
                if self.usage["depth"]:
                    loss += torch.abs(outputs[f'{prefix}_depth'] - targets['depth']).mean()
                if self.usage["gradient"]:
                    loss += img2mse(outputs[f'{prefix}_opacity'] - targets['opacity'])

        
        sparsity_loss = self.args.sparse_loss_weight \
            * (outputs["coarse_sparsity_loss"].sum() \
            + outputs.get("fine_sparsity_loss", torch.tensor(0)).sum())
        loss += sparsity_loss

        # add Total Variation loss
        if self.embedder_config["xyz"]["type"] == "hash":
            em = self.embedders["xyz"]
            TV_loss = sum(total_variation_loss(em, i) for i in range(em.n_levels))
            loss += self.args.tv_loss_weight * TV_loss
            if iter > 1000:
                self.args.tv_loss_weight = 0.0

        loss.backward()
        self.optimizer.step()

        return loss, psnr

    def update_lr(self):
        """Update the learning rate after an iteration"""
        decay_steps = self.lr_decay * 1000
        new_lr = self.lr * (self.decay_rate ** (self.global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save_ckpt(self, iter):
        """Save checkpoint"""    
        path = os.path.join(self.savepath, f'{iter:06d}.tar')

        ckpt_dict = {"global_step": self.global_step}
        # models
        for k, v in self.models.items():
            ckpt_dict["model"][k] = v.state_dict()
        # embedders (only if trainable)
        for k, v in self.embedders.items():
            if any(p for p in v.parameters() if p.requires_grad):
                ckpt_dict["embedder"][k] = v.state_dict()
        # optimizer
        ckpt_dict["optimizer"] = self.optimizer.state_dict()
        torch.save(ckpt_dict, path)
        print('Saved checkpoints at', path)
            
    def eval_test(self, test_fp):
        """
        Testing set evaluation.
        """
        os.makedirs(test_fp, exist_ok=True)
        with torch.no_grad():
            rgbs, _ = self.render_save(self.rays_test.o, self.rays_test.d, 
                                        savepath=test_fp)
        print('Done rendering', test_fp)

        # calculate MSE and PSNR for last image(gt pose)
        gt_loss = img2mse(torch.tensor(rgbs[-1]), torch.tensor(self.rays_test.rgb[-1]))
        gt_psnr = mse2psnr(gt_loss)
        pstr = f"GT Loss: {gt_loss}, PSNR: {gt_psnr}"
        print(pstr)
        with open(os.path.join(test_fp, 'stats.txt'), 'w') as f:
            f.write(pstr)

        rgbs = np.concatenate([rgbs[:-1],rgbs[:-1][::-1]])
        imageio.mimwrite(os.path.join(test_fp, 'video.gif'), to_8b(rgbs), duration=1000//10)
        print('Saved test set')   

    def render_save(self, rays_o, rays_d, savepath):
        """
        Render to save path

        func for eval_test
        """
        rgbs = []
        depths = []

        h, w = self.cc.h, self.cc.w  
        batch = h * w
        for idx in trange(rays_o.shape[0] // batch):
            start = idx * batch
            end = (idx + 1) * batch
            rays, og_shape = \
                prepare_rays(self.cc, rays=[rays_o[start:end], rays_d[start:end]], 
                             ndc=False, use_viewdirs=self.usage["dir"])

            outputs = self.volren.render(rays=rays)
            outputs = {k: torch.reshape(v, og_shape + list(v.shape[1:])) for k, v in outputs.items()}

            prefix = "fine" if self.args.N_fine > 0 else "coarse"
            rgb, depth = outputs[f"{prefix}_rgb"], outputs[f"{prefix}_depth"]
            if idx == 0:
                print(rgb.shape, depth.shape)
            rgb = rgb.reshape(h, w, 3).cpu().numpy()
            rgbs.append(rgb)

            depth = depth.reshape(h, w).cpu().numpy()
            depths.append(depth)

            if savepath is not None:
                save_imgs(rgb, depth, idx, savepath)

        rgbs = np.stack(rgbs, 0)
        depths = np.stack(depths, 0)
        return rgbs, depths 
