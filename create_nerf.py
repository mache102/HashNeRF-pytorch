import os
import torch
import torch.nn as nn
from radam import RAdam

from embedding.embedder import Embedder
from embedding.hash_encoding import HashEmbedder 
from embedding.spherical_harmonic import SHEncoder
from models import NeRF, NeRFSmall, NeRFGradient

from run_nerf_helpers import run_network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_nerf(args):
    """
    Instantiate NeRF's MLP model, among other things.
    """

    """
    step 1: create embedding functions

    TODO: how to create embed for equirect + hash?
    """
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    """
    Step 2: create coarse NeRF model
    """
    if args.i_embed==1:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        
    elif not args.use_gradient:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    else:
        model = NeRFGradient(D=args.netdepth, W=args.netwidth,
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        

    grad_vars = list(model.parameters())

    """
    Step 3: create fine NeRF model
    """
    model_fine = None

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

        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                net_chunk=args.net_chunk)

    """
    Step 4: create optimizer
    """
    # Create optimizer
    if args.i_embed==1:
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    """
    Step 5: Load checkpoints if available
    """
    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    # pdb.set_trace()

    """
    Step 6:
    Batch arguments and return
    """
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if (args.dataset_type not in ['llff', 'equirect']) or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


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
