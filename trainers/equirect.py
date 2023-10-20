import argparse
import torch
import numpy as np

import imageio

from tqdm import trange, tqdm 

from data_classes import EquirectDataset
from renderer import VolumetricRenderer
from renderer_util import prepare_rays
from ray_util import * 
from util import *

from trainers.base import BaseTrainer 

device = "cuda" if torch.cuda.is_available() else "cpu"

class EquirectTrainer(BaseTrainer):
    def __init__(self, dataset: EquirectDataset, models: dict, 
                 optimizer: torch.optim.Optimizer,
                 embedders: dict, args: argparse.Namespace):

        self.args = args
        self.models = models
        self.optimizer = optimizer
        self.embedders = embedders

        self.unpack_dataset(dataset)
        self.volren = VolumetricRenderer(cc=self.cc, models=models,
                                         embedders=embedders,
                                         args=args)

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
        self.rays_train = shuffle_rays(all_to_tensor(dataset.rays_train, device))
        self.rays_test = all_to_tensor(dataset.rays_test, device)
        self.bbox = dataset.bbox

        self.unpack_cc(dataset.cc)

    def fit(self):
        """
        Training loop
        """
        print(f"Training iterations {self.start} to {self.end}")

        for iter in trange(self.start, self.end):
            # retrieve batch of data
            batch, targets = self.get_batch(start=self.next_batch_idx, step=self.train_bsz)
            self.next_batch_idx += self.train_bsz

            # shuffle 
            if self.next_batch_idx >= self.rays_train.rgb.shape[0]:
                print("Shuffle data after an epoch!")
                self.rays_train = shuffle_rays(self.rays_train)
                self.next_batch_idx = 0 # reset

            # optimize
            rays, reshape_to = prepare_rays(self.cc, rays=[batch["o"], batch["d"]], 
                                            ndc=self.args.ndc, use_viewdirs=self.args.use_viewdirs)
            preds, extras = self.volren.render(rays=rays, reshape_to=reshape_to)
            loss, psnr, psnr_0 = self.calc_loss(preds, targets, extras)
            self.update_lr()
            self.log_progress(iter)

            if iter % self.args.i_print == 0:
                tqdm.write(f"[TRAIN] Iter: {iter} Loss: {loss.item()}  PSNR: {psnr.item()}")

            self.global_step += 1


    def get_batch(self, start, step):
        end = start + step   
        batch = {
            "o": self.rays_train.o[start:end],
            "d": self.rays_train.d[start:end]
        }

        target = {
            "rgb_map": self.rays_train.rgb[start:end],
            "depth_map": self.rays_train.d[start:end], # d not depth?
            "accumulation_map": self.rays_train.gradient[start:end]
        }
        return batch, target

    def calc_loss(self, preds, targets, extras):
        
        # unpack
        rgb = preds["rgb_map"]
        depth = preds["depth_map"]
        gradient = preds["accumulation_map"]

        t_rgb = targets["rgb_map"]
        t_depth = targets["depth_map"]
        t_gradient = targets["accumulation_map"]

        # final psnr
        # print(rgb.shape, t_rgb.shape)
        rgb_loss = img2mse(rgb, t_rgb)
        psnr = mse2psnr(rgb_loss)
        loss = rgb_loss
        # depth_loss
        if self.args.use_depth:
            loss += torch.abs(depth - t_depth).mean()
            
        if self.args.use_gradient:
            loss += img2mse(gradient, t_gradient)
        
        # coarse psnr (if coarse is not final)
        psnr_0 = None
        if 'rgb_0' in extras:
            rgb_loss_0 = img2mse(extras['rgb_0'], t_rgb)
            loss += rgb_loss_0
            if self.args.use_depth:
                loss += torch.abs(extras['depth_0'] - t_depth).mean()

            if self.args.use_gradient:
                loss += img2mse(extras['grad_0'], t_gradient)

            psnr_0 = mse2psnr(rgb_loss_0)

        loss.backward()
        self.optimizer.step()

        return loss, psnr, psnr_0

    def update_lr(self):
        decay_steps = self.args.lr_decay * 1000
        new_lr = self.args.lr * (self.decay_rate ** (self.global_step / decay_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def log_progress(self, iter):
        """
        SAVE CHECKPOINT
        """
        if iter % self.args.i_weights == 0:
            path = os.path.join(self.savepath, '{:06d}.tar'.format(iter))
            if self.args.i_embed == "hash":
                torch.save({
                    'global_step': self.global_step,
                    'coarse_model_state_dict': self.models["coarse"].state_dict(),
                    'fine_model_state_dict': self.models["fine"].state_dict(),
                    'pos_embedder_state_dict': self.embedders["pos"].state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': self.global_step,
                    'coarse_model_state_dict': self.models["coarse"].state_dict(),
                    'fine_model_state_dict': self.models["fine"].state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        
        if iter % self.args.i_testset == 0 and iter > 0:
            if self.args.stage > 0:
                test_fp = os.path.join(self.savepath, 'stage{}_test_{:06d}'.format(self.args.stage, iter))
            else:
                test_fp = os.path.join(self.savepath, 'testset_{:06d}'.format(iter))

            self.eval_test(test_fp)
            
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
        print('ground truth loss: {}, psnr: {}'.format(gt_loss, gt_psnr))
        with open(os.path.join(test_fp, 'statistics.txt'), 'w') as f:
            f.write('loss: {}, psnr: {}'.format(gt_loss, gt_psnr))

        rgbs = np.concatenate([rgbs[:-1],rgbs[:-1][::-1]])
        imageio.mimwrite(os.path.join(test_fp, 'video2.gif'), to_8b(rgbs), duration=1000//10)
        print('Saved test set')   

    def render_save(self, rays_o, rays_d, savepath):
        """
        Render to save path

        func for eval_test
        """
        rgbs = []
        depths = []

        temp_h = self.h 
        temp_w = self.w

        h = self.h * self.args.render_factor 
        w = self.w * self.args.render_factor


        batch = h * w    
        for idx in trange(rays_o.shape[0] // batch):
            start = idx * batch
            end = (idx + 1) * batch
            rays, reshape_to = \
                self.prepare_rays(self.cc, 
                                  rays=[rays_o[start:end], rays_d[start:end]], 
                                  ndc=self.args.ndc)
            rgb, depth, _, _ = \
                self.volren.render(rays=rays, reshape_to=reshape_to)
            
            if idx == 0:
                print(rgb.shape, depth.shape)
            rgb = rgb.reshape(h, w, 3).cpu().numpy()
            rgbs.append(rgb)

            depth = depth.reshape(h, w).cpu().numpy()
            depths.append(depth)

            if savepath is not None:
                save_imgs(rgb, depth, idx, savepath)

        # revert the height and width after rendering
        self.volren.h = temp_h 
        self.volren.w = temp_w

        rgbs = np.stack(rgbs, 0)
        depths = np.stack(depths, 0)
        return rgbs, depths 
