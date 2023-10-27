import argparse
import torch
import numpy as np
import os 
import imageio

from dataclasses import dataclass 
from tqdm import trange, tqdm 

from load_data import EquirectDataset
from renderer import VolumetricRenderer
from util.util import all_to_tensor, save_imgs, shuffle_rays
from util.math import to_8b, img2mse, mse2psnr


device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer():
    def __init__(self, dataset: EquirectDataset, models: dict, 
                 optimizer: torch.optim.Optimizer,
                 embedders: dict, usage: dict, 
                 args: argparse.Namespace):

        self.args = args
        self.models = models
        self.optimizer = optimizer
        self.embedders = embedders
        self.usage = usage

        self.unpack_dataset(dataset)
        self.volren = VolumetricRenderer(models=models,
                                         embedders=embedders,
                                         usage=usage, args=args)

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
        self.decay_rate = 0.1

    def unpack_dataset(self, dataset):
        """
        unpack train&test rays, bbox, and camera config
        """
        self.cc = dataset.cc
        self.rays_train = shuffle_rays(all_to_tensor(dataset.rays_train, device))
        self.rays_test = all_to_tensor(dataset.rays_test, device)
        self.bbox = dataset.bbox

    def fit(self):
        """
        Training loop
        """
        print(f"Training iterations {self.start} to {self.end}")

        for iteration in trange(self.start, self.end):
            self.iteration = iteration 

            # retrieve batch of data
            rays, targets = self.get_batch(start=self.next_batch_idx, step=self.train_bsz)
            self.next_batch_idx += self.train_bsz

            # shuffle 
            if self.next_batch_idx >= self.rays_train.color.shape[0]:
                print("Shuffle data after an epoch!")
                self.rays_train = shuffle_rays(self.rays_train)
                self.next_batch_idx = 0 # reset

            outputs = self.volren.render(rays=rays)
            loss, psnr = self.calc_loss(outputs, targets)
            self.update_lr()

            if iteration % self.args.iter_ckpt == 0:
                self.save_ckpt()

            if iteration % self.args.iter_test == 0 and iteration > 0:
                if self.args.stage > 0:
                    test_fp = os.path.join(self.savepath, 'stage{}_test_{:06d}'.format(self.args.stage, iteration))
                else:
                    test_fp = os.path.join(self.savepath, 'testset_{:06d}'.format(iteration))

                self.eval_test(test_fp)
                

            if iteration % self.args.iter_print == 0:
                tqdm.write(f"[TRAIN] Iter: {iteration} Loss: {loss.item()}  PSNR: {psnr.item()}")

            self.global_step += 1


    def get_batch(self, start, step, get_target=True):
        """
        Retrieve a batch of rays (self.train_bsz)
        for training 
        """
        end = start + step   
        dir = self.rays_train.direction[start:end]

        # (train_bsz, 3+3+1+1)
        rays = torch.cat([self.rays_train.origin[start:end], 
                          dir,
                          torch.full((step, 1), self.cc.near),
                          torch.full((step, 1), self.cc.far)], -1)

        if self.usage["dir"]:
            dir = dir / torch.norm(dir, dim=-1, keepdim=True)
            rays = torch.cat([rays, dir], -1)
   
        rays = torch.cat([rays, self.rays_train.ts[start:end]], -1)

        if not get_target:
            return rays

        gradient = self.rays_train.gradient[start:end] if self.usage["gradient"] else None
        targets = {
            "color": self.rays_train.color[start:end],
            "depth": self.rays_train.depth[start:end],
            "gradient": gradient,
        }
    
        return rays, targets

    def calc_loss(self, outputs, targets):
        self.optimizer.zero_grad()

        loss_dict = {}
        if self.args.N_fine > 0:
            loss_dict['color'] = img2mse(outputs["fine_color"], targets["color"])
            psnr = mse2psnr(loss_dict['color'])

            if self.usage["depth"]:
                loss_dict['depth'] = torch.abs(outputs["fine_depth"] - targets["depth"]).mean() 
                loss_dict['depth_'] = torch.abs(outputs["coarse_depth"], targets["depth"]).mean()

            if self.usage["gradient"]:
                loss_dict['gradient'] = img2mse(outputs["fine_gradient"], targets["gradient"])
                loss_dict['gradient_'] = img2mse(outputs["coarse_gradient"], targets["gradient"])
        
        else:
            loss_dict['color'] = img2mse(outputs["coarse_color"], targets["color"])
            psnr = mse2psnr(loss_dict['color'])

            if self.usage["depth"]:
                loss_dict['depth'] = torch.abs(outputs["coarse_depth"], targets["depth"]).mean()

            if self.usage["gradient"]:
                loss_dict['gradient'] = img2mse(outputs["coarse_gradient"], targets["gradient"])
               
        loss = sum(loss_dict.values())

        # sparsity_loss = self.args.sparse_loss_weight \
        #     * (outputs["coarse_sparsity_loss"].sum() \
        #     + outputs.get("fine_sparsity_loss", torch.tensor(0)).sum())
        # loss += sparsity_loss


        loss.backward()
        self.optimizer.step()

        return loss, psnr

    def update_lr(self):
        decay_steps = self.args.lr_decay * 1000
        new_lr = self.args.lr * (self.decay_rate ** (self.global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def save_ckpt(self):
        """Save checkpoint"""    
        path = os.path.join(self.savepath, f'{self.iteration:06d}.tar')

        ckpt_dict = {
            "global_step": self.global_step,
            "model": {},
            "embedder": {},
        }
        # models
        for k, v in self.models.items():
            if v is None: continue
            ckpt_dict["model"][k] = v.state_dict()
        # embedders (only if trainable)
        for k, v in self.embedders.items():
            if v is None: continue
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
            rgbs, _ = self.render_save(self.rays_test.origin, savepath=test_fp)
        print('Done rendering', test_fp)

        # calculate MSE and PSNR for last image(gt pose)
        gt_loss = img2mse(torch.tensor(rgbs[-1]), torch.tensor(self.rays_test.color[-1]))
        gt_psnr = mse2psnr(gt_loss)
        print('ground truth loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
        with open(os.path.join(test_fp, 'statistics.txt'), 'w') as f:
            f.write('loss: {}, psnr: {}'.format(gt_loss, gt_psnr))

        rgbs = np.concatenate([rgbs[:-1],rgbs[:-1][::-1]])
        imageio.mimwrite(os.path.join(test_fp, 'video2.gif'), to_8b(rgbs), duration=1000//10)
        print('Saved test set')   

    def render_save(self, rays_o, savepath):
        """
        Render to save path

        func for eval_test
        """
        rgbs = []
        depths = []

        h = self.cc.height * self.args.render_factor 
        w = self.cc.width * self.args.render_factor

        batch = h * w    
        for idx in trange(rays_o.shape[0] // batch):
            start = idx * batch

            rays = self.get_batch(start, step=batch, get_target=False)

            outputs = self.volren.render(rays=rays)
            prefix = "fine" if self.args.N_fine > 0 else "coarse"
            rgb, depth = outputs[f"{prefix}_color"], outputs[f"{prefix}_depth"]

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
