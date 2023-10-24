import torch 

from .hash_nerf import HashNeRF
from .vanilla_nerf import VanillaNeRF
from .nerfw import NeRFW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_networks(model_config, input_chs, use, args):
    """
    Initialize the coarse and fine networks.
    """
    models = {
        "coarse": None,
        "fine": None
    }

    coarse_config = model_config["coarse"]
    coarse_type = coarse_config["type"]
    if use["fine"]: 
        fine_config = model_config["fine"]
        fine_type = fine_config["type"]

    # coarse model
    if coarse_type == "vanilla_nerf":
        if use["fine"]:
            coarse_config["output_ch"] += 1
            fine_config["output_ch"] += 1
        models["coarse"] = VanillaNeRF(model_config["coarse"], input_chs=input_chs, 
                                        use_gradient=use["gradient"],
                                        use_viewdirs=use["dirs"])
        print("COARSE model: VanillaNeRF")

    elif coarse_type == "hash_nerf":
        models["coarse"] = HashNeRF(model_config["coarse"], 
                                    input_chs=input_chs)
        print("COARSE model: HashNeRF")

    # fine model
    if args.N_fine > 0:
        if fine_type == "vanilla_nerf":
            models["fine"] = VanillaNeRF(model_config["fine"], input_chs=input_chs, 
                                         use_gradient=use["gradient"],
                                         use_viewdirs=use["dirs"])
            print("FINE model: VanillaNeRF")
        elif fine_type == "hash_nerf":
            models["fine"] = HashNeRF(model_config["fine"], 
                                  input_chs=input_chs)
            print("FINe model: HashNeRF")
            
        elif fine_type == "nerfw":
            models["fine"] = NeRFW(model_config["fine"], input_chs=input_chs,
                                   use_appearance=use["appearance"],
                                   use_transient=use["transient"])
            print("Fine model: VanillaNeRF")

    return models
        
