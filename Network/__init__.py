import torch 

from .hash_nerf import HashNeRF
from .vanilla_nerf import VanillaNeRF
from .nerfw import NeRFW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_networks(model_config, input_chs, args):
    """
    Initialize the coarse and fine networks.
    """
    models = {
        "coarse": None,
        "fine": None
    }

    use_fine = model_config.get("fine") is not None
    coarse_config = model_config["coarse"]
    coarse_type = coarse_config["type"]
    if use_fine: 
        fine_config = model_config["fine"]
        fine_type = fine_config["type"]

    # coarse model
    if coarse_type == "vanilla_nerf":
        if use_fine:
            coarse_config["output_ch"] += 1
            fine_config["output_ch"] += 1
        models["coarse"] = VanillaNeRF(model_config["coarse"], input_chs=input_chs, 
                                        use_viewdirs=args.use_viewdirs,
                                        use_gradient=args.use_gradient)
        print("COARSE model: VanillaNeRF")

    elif coarse_type == "hash_nerf":
        models["coarse"] = HashNeRF(model_config["coarse"], 
                                    input_chs=input_chs)
        print("COARSE model: HashNeRF")

    # fine model
    if args.fine_samples > 0:
        if fine_type == "vanilla_nerf":
            models["fine"] = VanillaNeRF(model_config["fine"], input_chs=input_chs, 
                                    use_viewdirs=args.use_viewdirs,
                                    use_gradient=args.use_gradient)
            print("FINE model: VanillaNeRF")
        elif fine_type == "hash_nerf":
            models["fine"] = HashNeRF(model_config["fine"], 
                                  input_chs=input_chs)
            print("FINe model: HashNeRF")
            
        elif fine_type == "nerfw":
            models["fine"] = NeRFW(model_config["fine"], input_chs=input_chs,
                               use_appearance=fine_config.get("appearance") is not None,
                               use_transient=fine_config.get("transient") is not None)
            print("Fine model: VanillaNeRF")

    return models
        
if __name__ == '__main__':
    from settings import NET_CONFIG_PATH
    import json
    import os 

    fp = os.path.join(NET_CONFIG_PATH, f"hash_nerf.json")
    if not os.path.isfile(fp):
        raise ValueError(f"Model configuration not found: {fp}")
    with open(fp, "r") as f:
        model_config = json.load(f)
    models_ = get_networks(model_config, input_chs, args)
    models.update(models_)
    