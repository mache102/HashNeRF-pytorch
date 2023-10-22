from torch.optim import Adam
from radam import RAdam 

def get_optimizer(otyp, grad_vars, args, em_xyz_params=None):
    if otyp == "radam":      
        optimizer = \
            RAdam([
                {'params': grad_vars, 'weight_decay': 1e-6},
                {'params': em_xyz_params, 'eps': 1e-15}
            ], lr=args.lr, betas=(0.9, 0.99))
        print("Optimizer: RAdam")

    elif otyp == "adam":
        optimizer = \
            Adam(
                params=grad_vars, 
                lr=args.lr, 
                betas=(0.9, 0.999)
            )
        print("Optimizer: Adam")
    else: 
        raise NotImplementedError

    return optimizer