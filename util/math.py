import torch
import numpy as np 


def to_8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def psnr(pred_img, gt_img):
    return -10. * np.log10(np.mean(np.square(pred_img - gt_img)))

def img2mse(x, y):
    return torch.mean((x - y) ** 2)

def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
