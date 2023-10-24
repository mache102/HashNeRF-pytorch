import os 
import torch  

class TrainHelper:
    def __init__(self, models: dict, embedders: dict):
        self.models = models 
        self.embedders = embedders 
        pass 

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