import argparse
import torch
import numpy as np
import time 
import pickle
import imageio

from tqdm import trange , tqdm 

from loss import sigma_sparsity_loss, total_variation_loss
from load.load_data import StandardDataset
from Renderer.renderer import *
from ray_util import * 
from util import *
from math_util import to_8b, img2mse, mse2psnr, psnr

from trainers.base import BaseTrainer 

device = "cuda" if torch.cuda.is_available() else "cpu"

class StandardTrainer(BaseTrainer):
    def __init__(self, dataset: StandardDataset, start,
                 models: dict, optimizer: torch.optim.Optimizer,
                 embedders: dict, model_config: dict, embedder_config: dict,
                 args: argparse.Namespace):

        self.args = args
        self.models = models
        self.optimizer = optimizer
        self.embedders = embedders
        self.model_config = model_config
        self.embedder_config = embedder_config
        self.use_viewdirs = embedder_config.get("viewdirs") is not None

        self.unpack_dataset(dataset)
        self.volren = VolumetricRenderer(cc=self.cc, models=models,
                                         embedders=embedders,
                                         args=args)

        self.savepath = args.savepath
        self.bsz = args.train_bsz
        self.iters = args.train_iters + 1
        self.use_batching = args.use_batching

        self.precrop = {
            "iters": args.precrop_iters,
            "frac": args.precrop_frac
        }

        if self.use_batching:
            # For random ray batching
            print("Using random ray batching")

            rays = np.stack([get_rays_np(self.h, self.w, self.k, p) for p in self.poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
            print("Concatenated direction and origin rays")
            rays_rgb = np.concatenate([rays, self.images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in self.i_train], 0) # train images only
            rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)

            np.random.shuffle(rays_rgb)
            rays_rgb = torch.Tensor(rays_rgb).to(device)
            print("Shuffled rays")

            self.i_batch = 0

        poses = torch.Tensor(poses).to(device)

        self.loss_list = []
        self.psnr_list = []
        self.time_list = []
        
        self.global_step = start
        self.start = start + 1

    def unpack_dataset(self, dataset):
        self.i_train, self.i_val, self.i_test = \
            dataset.train, dataset.val, dataset.test
        
        print(f"Train views: {self.i_train}\n"
              f"Val views: {self.i_val}\n"
              f"Test views: {self.i_test}")
        
        self.images = torch.Tensor(dataset.images).to(device)
        self.poses = dataset.poses
        self.render_poses = dataset.render_poses
        if self.args.render_test:
            self.render_poses = np.array(self.poses[self.i_test])
        self.render_poses = torch.Tensor(self.render_poses).to(device)
    
        self.unpack_cc(dataset.cc)

    def fit(self):
        for iter in trange(self.start, self.iters):
            """
            Training iteration
            """
            t1 = time.time()
            batch_rays, target_s = self.get_batch(iter)
            # optimizer
            rays, og_shape = prepare_rays(self.cc, rays=batch_rays, ndc=self.args.ndc,
                                            use_viewdirs=self.use_viewdirs)
            rgb, depth, opacity, extras = \
                self.volren.render(rays=rays, og_shape=og_shape,
                                   verbose=iter < 10, retraw=True)
            self.optimizer.zero_grad()

            loss, psnr = self.calc_loss(iter, rgb, target_s, extras)
            # update lr
            decay_rate = 0.1
            decay_steps = self.args.lr_decay * 1000
            new_lr = self.args.lr * (decay_rate ** (self.global_step / decay_steps))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            t = time.time() - t1

            self.log_progress(iter, t, loss, psnr)
            self.global_step += 1

    def get_batch_all_imgs(self):
        start, end = self.i_batch, self.i_batch + self.bsz
        # Random over all images
        batch = self.rays_rgb[start:end] # [B, 2+1, 3*?]
        batch = batch.transpose(0, 1)
        batch_rays, target_s = batch[:2], batch[2]

        self.i_batch += self.bsz
        if self.i_batch >= self.rays_rgb.shape[0]:
            print("Shuffle data after an epoch")
            rand_idx = torch.randperm(self.rays_rgb.shape[0])
            self.rays_rgb = self.rays_rgb[rand_idx]
            self.i_batch = 0
        
        return batch_rays, target_s
    
    def get_batch_one_img(self, iter):
        # Random from one image
        i_img = np.random.choice(self.i_train)
        image = torch.Tensor(self.images[i_img]).to(device)
        pose = self.poses[i_img, :3,:4]

        if self.bsz is not None:
            rays_o, rays_d = get_rays(self.h, self.w, self.k, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if iter < self.precrop["iters"]:
                dh = int(self.h // 2 * self.precrop["frac"])
                dw = int(self.w // 2 * self.precrop["frac"])
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(self.h // 2 - dh, self.h // 2 + dh - 1, 2 * dh),
                        torch.linspace(self.w // 2 - dw, self.w // 2 + dw - 1, 2 * dw)
                    ), -1)
                if iter == self.start:
                    print(f"[Config] Center cropping of size {2*dh} x {2*dw} is enabled until iter {self.precrop['iters']}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[self.bsz], replace=False)  # (bsz,)
            select_coords = coords[select_inds].long()  # (bsz, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (bsz, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (bsz, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0)
            target_s = image[select_coords[:, 0], select_coords[:, 1]]  # (bsz, 3)

        return batch_rays, target_s

    def get_batch(self, iter):
        if self.use_batching:
            batch_rays, target_s = self.get_batch_all_imgs()

        else:
            batch_rays, target_s = self.get_batch_one_img()

    def calc_loss(self, iter, rgb, target_s, extras,):
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if "rgb_map_0" in extras.keys():
            img_loss_0 = img2mse(extras['rgb_map_0'], target_s)
            loss += img_loss_0
            psnr_0 = mse2psnr(img_loss_0)

        sparsity_loss = self.args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss_0"].sum())
        loss += sparsity_loss

        # add Total Variation loss
        if self.args.em_xyz == "hash":
            em = self.embedders["xyz"]
            TV_loss = sum(total_variation_loss(em, i) for i in range(em.n_levels))
            loss += self.args.tv_loss_weight * TV_loss
            if iter > 1000:
                self.args.tv_loss_weight = 0.0

        loss.backward()
        self.optimizer.step()

        return loss, psnr

    def log_progress(self, iter, t, loss, psnr):
        """
        SAVE CHECKPOINT
        """
        if iter % self.args.iter_ckpt == 0:
            path = os.path.join(self.savepath, '{:06d}.tar'.format(iter))
            if self.args.em_xyz == "hash":
                torch.save({
                    'global_step': self.global_step,
                    'coarse_model_state_dict': self.models["coarse"].state_dict(),
                    'fine_model_state_dict': self.models["fine"].state_dict(),
                    'xyz_embedder_state_dict': self.embedders["xyz"].state_dict(),
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

        """
        RENDER VIDEO
        """
        if iter % self.args.iter_video == 0 and iter > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = self.render_save(poses=self.render_poses)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(self.savepath, '{}_spiral_{:06d}_'.format(self.args.expname, iter))
            imageio.mimwrite(moviebase + 'rgb.mp4', to_8b(rgbs), duration=1000//30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to_8b(disps / np.max(disps)), duration=1000//30, quality=8)

        """
        RENDER TEST SET
        """
        if iter % self.args.iter_test == 0 and iter > 0:
            test_fp = os.path.join(self.savepath, 'testset_{:06d}'.format(iter))
            self.eval_test(test_fp)

        """
        PRINT TRAINING PROGRESS
        """
        if iter % self.args.iter_print == 0:
            tqdm.write(f"[TRAIN] Iter: {iter} Loss: {loss.item()}  PSNR: {psnr.item()}")
            self.loss_list.append(loss.item())
            self.psnr_list.append(psnr.item())
            self.time_list.append(t)
            loss_psnr_time = {
                "losses": self.loss_list,
                "psnr": self.psnr_list,
                "time": self.time_list
            }
            with open(os.path.join(self.savepath, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)

    def eval_test(self, test_fp):
        os.makedirs(test_fp, exist_ok=True)
        if self.args.render_test:
            # render_test switches to test poses
            images = self.images[self.i_test]
            poses = torch.Tensor(self.poses[self.i_test]).to(device)
        else:
            # Default is smoother render_poses path
            images = None
            poses = self.render_poses
        print('test poses shape', self.render_poses.shape)

        # torch.Tensor(self.poses[self.i_test]).to(device) as poses
        with torch.no_grad():
            rgbs, _ = self.render_path(poses=poses, gt_imgs=images, savepath=test_fp, 
                                        render_factor=self.args.render_factor)
        print('Done rendering', test_fp)
        imageio.mimwrite(os.path.join(test_fp, 'video.mp4'), to_8b(rgbs), duration=1000//30, quality=8)


    def render_save(self, poses=None, gt_imgs=None, 
                    savepath=None,
                    render_factor=1, test=True):
        """
        Rendering for test set
        """
        if savepath is None:
            return 
        
        if poses is None:
            poses = self.render_poses
        rgbs = []
        depths = []
        psnrs = []

        temp_h = self.volren.h 
        temp_w = self.volren.w

        self.volren.h *= render_factor 
        self.volren.w *= render_factor
        for idx in trange(len(poses)):
            c2w = poses[idx]

            rgb, depth, _, _ = \
                self.volren.render(*prepare_rays(cc=self.cc, 
                                                 c2w=c2w[:3,:4], 
                                                 use_viewdirs=self.use_viewdirs))
            if idx == 0:
                print(rgb.shape, depth.shape)

            rgb = rgb.cpu().numpy()
            rgbs.append(rgb)
            depth = ((depth - self.near) / (self.far - self.near)).cpu().numpy()
            depths.append(depth)

            if gt_imgs is not None and render_factor == 1:
                try:
                    gt_img = gt_imgs[idx].cpu().numpy()
                except:
                    gt_img = gt_imgs[idx]
                
                p = psnr(rgb, gt_img)
                psnrs.append(p)

            save_imgs(rgb, depth, idx, savepath)         

        rgbs = np.stack(rgbs, 0)
        depths = np.stack(depths, 0)

        if gt_imgs is not None and render_factor == 1:
            avg_psnr = sum(psnrs)/len(psnrs)
            print("Avg PSNR over Test set: ", avg_psnr)
            with open(os.path.join(self.args.savepath, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
                pickle.dump(psnrs, fp)

        # revert the height and width after rendering
        self.volren.h = temp_h 
        self.volren.w = temp_w

        return rgbs, depths