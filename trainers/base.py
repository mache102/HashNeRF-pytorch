
"""
BaseModel for training 
"""
from tqdm import trange
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self):
        pass
     
    def unpack_cc(self, cc):
        """
        Unpack camera configs

        h: int, Height
        w: int, Width
        f: float, Focal
        k: np.ndarray, Intrinsic matrix
        near: float, Near plane
        far: float, Far plane
        """
        self.cc = cc 
        self.h = cc.height
        self.w = cc.width
        self.focal = cc.focal
        self.k = cc.k
        self.near = cc.near
        self.far = cc.far
 
    def fit(self):
        """
        training loop
        """
        for i in trange(10):
            self.get_batch()
            self.shuffle()
            self.optimize()
            self.calc_loss()
            self.update_lr()
            self.log_progress()
        
    def get_batch(self):
        """
        retrieve a single batch of data
        """
        pass

    def shuffle(self):
        """
        shuffle data
        """
        pass

    def optimize(self):
        """
        pass data to model and optimize
        """
        pass

    def calc_loss(self):
        """
        calculate loss(es)
        """
        pass

    def update_lr(self):
        """
        update learning rate
        """
        pass

    def log_progress(self):
        """
        log progress
        (test set, video, metrics)
        """
        pass
