import torch
import numpy as np 

def to_8b(x):
    """
    Normalize a 2d array to 8b image format
    """
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)

def psnr(pred, gt):
    """
    Calculate PSNR between pred and ground truth
    """
    return -10. * np.log10(np.mean(np.square(pred, gt)))

def img2mse(x, y):
    """
    Calculate MSE between two images
    """
    return torch.mean((x - y) ** 2)

def img2mse_np(x, y):
    """
    Numpy ver of above (since np.mean() fails on torch tensors)
    """
    return np.mean((x - y) ** 2)

def mse2psnr(x):
    """
    MSE to PSNR
    """
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

# def ssim(x, y):
#     """
#     Structural similarity index measure of two images
#     """
#     raise NotImplementedError

# def dssim(x, y):
#     """
#     Differentiable SSIM
#     """
#     return 1 - ssim(x, y)