import torch 
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from load.load_equirect import EquirectRays

@dataclass
class CameraConfig:
    height: int
    width: int
    near: float
    far: float
    focal: Optional[float] = None
    k: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.k is None:
            self.k = np.array([
                [self.focal, 0, 0.5 * self.width],
                [0, self.focal, 0.5 * self.height],
                [0, 0, 1]
            ])

@dataclass 
class EquirectDataset:
    cc: CameraConfig
    rays_train: EquirectRays
    rays_test: EquirectRays
    bbox: Tuple[torch.Tensor, torch.Tensor] 

@dataclass 
class StandardDataset:
    # TODO: specify types
    cc: CameraConfig
    images: any
    poses: any
    render_poses: any
    bbox: any

    train: any
    val: any
    test: any
