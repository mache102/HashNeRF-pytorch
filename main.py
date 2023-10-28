import json 
import os 
import torch
import numpy as np
import pdb
import traceback 

from radam import RAdam
from load_data import load
from embedders import get_embedders
from networks import get_networks
from util.util import create_expname, save_configs
from parse_args import config_parser

from trainer import Trainer 

CONFIG_PATH = "configs/"
# 20231010 15:25
  
def main():
    """
    Main NeRF pipeline    
    """

    if args.N_fine > 0:
        print("=== Using fine model ===")
    if args.render_only:
        print("=== Rendering only ===")

    """
    1. Load dataset
    """
    print("Load data")
    dataset = load(args)
    args.bbox = dataset.bbox
    """
    2. Experiment savepath, savename, etc.
    """
    # Create log dir and copy the config file
    args.expname = create_expname(args)
    args.savepath = os.path.join(args.save_path, args.expname)
    print(f"Experiment name: {args.expname}\n"
          f"Save path: {args.savepath}")
    os.makedirs(args.savepath, exist_ok=True)
    save_configs(args)
    print("Saved configs")

    """
    3. Create embedding functions
    """
    fp = os.path.join(CONFIG_PATH, f"EMBED_{args.embed_config}.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Embed configuration not found: {fp}")
    with open(fp, "r") as f:
        embedder_config = json.load(f)
        print(f"Loaded embed config {args.embed_config}")

    # some embedders are trainable so we add them to a model dict in advance
    embedders = get_embedders(embedder_config, dataset)
    input_chs = {}
    for k in embedders:
        if embedders[k] is None:
            input_chs[k] = 0
            continue
        input_chs[k] = embedders[k].out_dim

    em_xyz_params = None
    if embedder_config["xyz"]["type"] == "hash":
        # hashed embedding table
        em_xyz_params = list(embedders["xyz"].parameters())
    print("Embedders created")

    """
    4. Create coarse and fine models
    """
    fp = os.path.join(CONFIG_PATH, f"MODEL_{args.model_config}.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Model configuration not found: {fp}")
    with open(fp, "r") as f:
        model_config = json.load(f)
        print(f"Loaded model config {args.model_config}")

    usage = {
        "dir": embedder_config.get("dir") is not None,
        "tv_loss": embedder_config["xyz"]["type"] == "hash",
        "depth": args.use_depth,
        "gradient": args.use_gradient,
        "fine": args.N_fine > 0
    }
    models = get_networks(model_config, input_chs, usage, args)
    
    grad_vars = []
    for k in models:
        # move to device first
        models[k] = models[k].to(device)
        grad_vars += list(models[k].parameters())
    # for k in embedders:
    #     if is_trainable[k]:
    #         embedders[k] = embedders[k].to(device)
    #         grad_vars += list(embedders[k].parameters())
    print("Models created")

    """
    5. Create optimizer
    """
    if embedder_config["xyz"]["type"] == "hash":
        print("Optimizer: RAdam")
        optimizer = \
            RAdam([
                {'params': grad_vars, 'weight_decay': 1e-6},
                {'params': em_xyz_params, 'eps': 1e-15}
            ], lr=args.lr, betas=(0.9, 0.99))
    else:
        print("Optimizer: Adam")
        optimizer = \
            torch.optim.Adam(
                params=grad_vars, 
                lr=args.lr, 
                betas=(0.9, 0.999)
            )

    """
    6. Load checkpoints if available
    """
    # update this 
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
        if args.em_xyz == "hash":
            embedders["xyz"].load_state_dict(ckpt['xyz_embedder_state_dict'])

    args.global_step = global_step
    """
    7. Create trainer
    """
    trainer = Trainer(dataset=dataset, models=models, 
                      optimizer=optimizer, embedders=embedders, 
                      usage=usage, args=args)

    if args.render_only:
        render_only(trainer)
        return 

    print("Start training")
    trainer.fit()

def render_only(trainer):
    print('RENDER ONLY')
    if args.stage > 0:
        test_fp = os.path.join(args.savepath, 'renderonly_stage_{}_{:06d}'.format(args.stage, args.global_step))
    else:
        test_fp = os.path.join(args.savepath, 'renderonly_train_{}_{:06d}'.format('test' if args.render_test else 'path', args.global_step))


    trainer.eval_test(test_fp)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    parser = config_parser()
    args = parser.parse_args()
    args.train_iters += 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    try:
        main()
    except Exception as e:
        traceback.print_exc()
        pdb.post_mortem()
