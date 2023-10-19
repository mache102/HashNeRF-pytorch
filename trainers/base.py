
"""
BaseModel for training 
"""
from tqdm import trange
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
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
        
    @abstractmethod
    def get_batch(self):
        """
        retrieve a single batch of data
        """
        pass

    @abstractmethod
    def shuffle(self):
        """
        shuffle data
        """
        pass

    @abstractmethod
    def optimize(self):
        """
        pass data to model and optimize
        """
        pass

    @abstractmethod
    def calc_loss(self):
        """
        calculate loss(es)
        """
        pass

    @abstractmethod
    def update_lr(self):
        """
        update learning rate
        """
        pass

    @abstractmethod
    def log_progress(self):
        """
        log progress
        (test set, video, metrics)
        """
        pass
