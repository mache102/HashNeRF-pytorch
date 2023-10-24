import json 
import os 
import torch

from Optimizer.radam import RAdam
from ray_util import *
from load.load_data import load_data


from Network import get_networks
from Embedder import get_embedders
from Optimizer import get_optimizer

from Render import Render, BatchRender
from Render.batch_render import BatchRender

from util import *
from parse_args import config_parser

from trainers.trainer import EquirectTrainer 
from trainers.standard import StandardTrainer
from settings import NET_CONFIG_PATH, EMBED_CONFIG_PATH
# 20231010 15:25



def main():
    if args.N_fine > 0:
        print("=== Using fine model ===")
    if args.render_only:
        print("=== Rendering only ===")

    """
    1. Load dataset
    """
    print("Load data")
    dataset = load_data(args)
    # o, d, rgb, gradient: (total_rays, 3)
    # depth: (total_rays,)
    """
    2. Experiment savepath, savename, etc.
    """
    # Create log dir and copy the config file
    args.expname = create_expname(args)
    args.savepath = os.path.join(args.basedir, args.expname)
    print(f"Experiment name: {args.expname}\n"
          f"Save path: {args.savepath}")
    os.makedirs(args.savepath, exist_ok=True)
    save_configs(args)
    print("Saved configs")

    """
    3. Create embedding functions
    """
    fp = os.path.join(EMBED_CONFIG_PATH, f"{args.embed_config}.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Embed configuration not found: {fp}")
    with open(fp, "r") as f:
        embedder_config = json.load(f)
        print(f"Loaded embed config {args.embed_config}")

    # some embedders are trainable so we add them to a model dict in advance
    models = {}
    embedders, is_trainable = get_embedders(embedder_config, dataset)
    input_chs = {}
    for k in embedders:
        if embedders[k] is None:
            input_chs[k] = 0
            continue
        input_chs[k] = embedders[k].out_dim
        if is_trainable[k]:
            models[k] = embedders[k]
            # remove from embedders dict
            embedders[k] = None

    em_xyz_params = None
    if embedder_config["xyz"]["type"] == "hash":
        # hashed embedding table
        em_xyz_params = list(embedders["xyz"].parameters())
    print("Embedders created")
 
    """
    4. Create coarse and fine models
    """
    fp = os.path.join(NET_CONFIG_PATH, f"{args.model_config}.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Model configuration not found: {fp}")
    with open(fp, "r") as f:
        model_config = json.load(f)
        print(f"Loaded model config {args.model_config}")

    usage = {
        "dirs": embedder_config.get("viewdirs") is not None,
        "depth": args.use_depth,
        "gradient": args.use_gradient,
        "fine": args.N_fine > 0
    }

    if not usage["fine"]:
        usage["appearance"] = False
        usage["transient"] = False
    else:
        usage["appearance"]: model_config["fine"].get("appearance") is not None
        usage["transient"]: model_config["fine"].get("transient") is not None
        
    models_ = get_networks(model_config, input_chs, usage, args)
    models.update(models_)
    
    grad_vars = []
    for k in models:
        # move to device first
        models[k] = models[k].to(device)
        grad_vars += list(models[k].parameters())
    print("Models created")

    """
    5. Create optimizer

    Nothing much to configure here atm
    (func for future expansion)
    """
    otyp = "radam" if embedder_config["xyz"]["type"] == "hash" else "adam"
    optimizer = get_optimizer(otyp, grad_vars, args, em_xyz_params)

    """
    6. Load checkpoints if available
    """
    # TODO: update this
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
    batch_render = BatchRender(cc=cc, models=models,
                                embedders=embedders, 
                                usage=usage, args=args,
                                bbox=dataset.bbox)
    ren = Render(bsz=args.render_bsz, 
                batch_render=batch_render)

    if args.dataset_type == "equirect":
        print("Using equirect trainer")
        tclass = EquirectTrainer 
    else:
        print("Using standard trainer")
        tclass = StandardTrainer

    trainer = tclass(dataset=dataset, models=models, 
                     optimizer=optimizer, embedders=embedders, 
                     usage=usage, args=args)

    """
    8a. Render/test/evaluate, no training
    """
    if args.render_only:
        render_only(trainer)
        return 
    
    """
    8b. Enter training loop
    """
    print("Start training")
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
    ndc_list = ['llff', 'equirect']
    args.ndc = (args.dataset_type in ndc_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    main()
