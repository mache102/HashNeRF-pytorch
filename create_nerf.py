import os
import torch
import torch.nn as nn
from radam import RAdam

from embedding.embedder import Embedder
from embedding.hash_encoding import HashEmbedder 
from embedding.spherical_harmonic import SHEncoder
from models import NeRF, NeRFSmall, NeRFGradient

def create_nerf(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {
        "coarse": None,
        "fine": None
    }
    embedders = {
        "pos": None,
        "dir": None
    }

    """
    step 1: create embedding functions
    """
    
    # input ch as in model input ch
    embedders["pos"], input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed == 1:
        # hashed embedding table
        pos_embedder_params = list(embedders["pos"].parameters())

    input_ch_views = 0
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embedders["dir"], input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    """
    Step 2: Create coarse and fine models
    """
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.i_embed==1:
        model_coarse = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        
    elif not args.use_gradient:
        model_coarse = NeRF(D=args.netdepth, W=args.netwidth,
                input_ch=input_ch, output_ch=output_ch, skips=skips,
                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    else:
        model_coarse = NeRFGradient(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        
    models["coarse"] = model_coarse
    grad_vars = list(models["coarse"].parameters())

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        elif not args.use_gradient:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRFGradient(D=args.netdepth_fine, W=args.netwidth_fine,
                        input_ch=input_ch, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

        models["fine"] = model_fine
        grad_vars += list(models["fine"].parameters())
        
    

    """
    Step 3: create optimizer
    """
    # Create optimizer
    if args.i_embed == 1:
        optimizer = \
            RAdam([
                {'params': grad_vars, 'weight_decay': 1e-6},
                {'params': pos_embedder_params, 'eps': 1e-15}
            ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = \
            torch.optim.Adam(
                params=grad_vars, 
                lr=args.lrate, 
                betas=(0.9, 0.999)
            )

    start = 0
    savepath = args.savepath

    """
    Step 5: Load checkpoints if available
    """
    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(savepath, f) for f in sorted(os.listdir(os.path.join(savepath))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        models["coarse"].load_state_dict(ckpt['network_fn_state_dict'])
        if models["fine"] is not None:
            models["fine"].load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embedders["pos"].load_state_dict(ckpt['pos_embedder_state_dict'])

    ##########################
    # pdb.set_trace()

    """
    Step 5:
    Batch arguments and return
    """

    # testing kwargs:
    # perturb=  False 
    # raw noise std = 0
    return models, embedders, start, grad_vars, optimizer


def get_embedder(multires, args, i=0):
    """
    -1: no embedding
    0: Standard positional encoding (Nerf, section 5.1)
    1: Hashed pos encoding (Instant-ngp)
    2: Spherical harmonic encoding (?)
    """
    if i == -1:
        return nn.Identity(), 3
    elif i == 0:
        embed_kwargs = {
                    'include_input' : True,
                    'input_dims' : 3,
                    'max_freq_log2' : multires-1,
                    'num_freqs' : multires,
                    'log_sampling' : True,
                    'periodic_fns' : [torch.sin, torch.cos],
        }
        
        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj : eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i == 1:
        embed = HashEmbedder(bounding_box=args.bounding_box, \
                            log2_hashmap_size=args.log2_hashmap_size, \
                            finest_resolution=args.finest_res)
        out_dim = embed.out_dim
    elif i == 2:
        embed = SHEncoder()
        out_dim = embed.out_dim
    return embed, out_dim
