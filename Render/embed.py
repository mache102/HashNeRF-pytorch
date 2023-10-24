import torch 

def embed_all(inputs, embedders):
    """
    Embed all the inputs.
    """
    embedded = torch.empty(0)  
    for k, v in embedders.items():
        out = v(inputs[k])
        embedded = torch.cat([embedded, out], -1)
    return embedded
