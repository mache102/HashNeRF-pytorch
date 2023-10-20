import json 
import os 
import torch

from networks.hash_nerf import HashNeRF
from networks.vanilla_nerf import VanillaNeRF
from radam import RAdam
from ray_util import *
from load.load_data import load_data

from util import *
from parse_args import config_parser

from trainers.equirect import EquirectTrainer 
from trainers.standard import StandardTrainer

# 20231010 15:25
  
def main():
    # load dataset
    dataset = load_data(args)
    """
    Experiment savepath, savename, etc.
    """
    # Create log dir and copy the config file
    args.expname = create_expname(args)
    args.savepath = os.path.join(args.basedir, args.expname)
    os.makedirs(args.savepath, exist_ok=True)
    save_configs(args)

    """
    Load model config
    """
    fp = os.path.join("model_configs", f"{args.model_config}.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Model configuration not found: {fp}")
    with open(fp, "r") as f:
        model_config = json.load(f)

    """
    Create embedding functions for position & viewdirs
    """
    embedders = {
        "pos": None,
        "dir": None
    }
    # input ch as in model input ch
    embedders["pos"], input_ch = get_embedder(name=args.i_embed, args=args, 
                                              multires=args.multires)
    if args.i_embed == 'hash':
        # hashed embedding table
        pos_embedder_params = list(embedders["pos"].parameters())

    input_ch_views = 0
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embedders["dir"], input_ch_views = get_embedder(name=args.i_embed_views,
                                                        args=args, multires=args.multires_views)

    """
    Create coarse and fine models
    """
    models = {
        "coarse": None,
        "fine": None
    }
    if args.i_embed == "hash":
        model_coarse = HashNeRF(model_config["coarse"], input_ch=input_ch, 
                                input_ch_views=input_ch_views).to(device)
        
    elif not args.use_gradient:
        if args.importance > 0:
            model_config["coarse"]["output_ch"] += 1
            model_config["fine"]["output_ch"] += 1
        model_coarse = VanillaNeRF(model_config["coarse"], input_ch=input_ch, 
                                   input_ch_views=input_ch_views, 
                                   use_viewdirs=args.use_viewdirs,
                                   use_gradient=args.use_gradient).to(device)
    models["coarse"] = model_coarse
    grad_vars = list(models["coarse"].parameters())

    if args.N_importance > 0:
        if args.i_embed == "hash":
            model_fine = HashNeRF(model_config["fine"], input_ch=input_ch, 
                                    input_ch_views=input_ch_views).to(device)
            
        elif not args.use_gradient:
            model_fine = VanillaNeRF(model_config["fine"], input_ch=input_ch, 
                                    input_ch_views=input_ch_views, 
                                    use_viewdirs=args.use_viewdirs,
                                    use_gradient=args.use_gradient).to(device)
        models["fine"] = model_fine
        grad_vars += list(models["fine"].parameters())

    """
    Create optimizer
    """
    if args.i_embed == "hash":
        optimizer = \
            RAdam([
                {'params': grad_vars, 'weight_decay': 1e-6},
                {'params': pos_embedder_params, 'eps': 1e-15}
            ], lr=args.lr, betas=(0.9, 0.99))
    else:
        optimizer = \
            torch.optim.Adam(
                params=grad_vars, 
                lr=args.lr, 
                betas=(0.9, 0.999)
            )

    """
    Load checkpoints if available
    """
    global_step = 0
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = []
        for f in sorted(os.listdir(args.savepath)):
            if "tar" in f:
                ckpts.append(os.path.join(args.savepath, f))

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and args.reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        global_step = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        models["coarse"].load_state_dict(ckpt['coarse_model_state_dict'])
        if models["fine"] is not None:
            models["fine"].load_state_dict(ckpt['fine_model_state_dict'])
        if args.i_embed == "hash":
            embedders["pos"].load_state_dict(ckpt['pos_embedder_state_dict'])

    args.global_step = global_step
    """
    Create trainer
    """
    if args.dataset_type == "equirect":
        tclass = EquirectTrainer 
    else:
        tclass = StandardTrainer
    trainer = tclass(dataset=dataset, models=models, 
                     optimizer=optimizer, embedders=embedders, 
                     args=args)

    """
    Skip to render only
    """
    if args.render_only:
        render_only(trainer)
        return 
    
    """
    Training loop
    """
    trainer.fit()

def render_only(trainer):
    print('RENDER ONLY')
    if args.dataset_type == 'equirect':
        if args.stage > 0:
            test_fp = os.path.join(args.savepath, 'renderonly_stage_{}_{:06d}'.format(args.stage, args.global_step))
        else:
            test_fp = os.path.join(args.savepath, 'renderonly_train_{}_{:06d}'.format('test' if args.render_test else 'path', args.global_step))
    else:
        test_fp = os.path.join(args.savepath, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', args.global_step))    

    trainer.eval_test(test_fp)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    args.train_iters += 1
    if args.dataset_type != 'llff':
        args.ndc = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    main()