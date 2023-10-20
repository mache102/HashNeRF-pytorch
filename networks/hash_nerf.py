import torch 
import torch.nn as nn

# Small NeRF for Hash embeddings
class HashNeRF(nn.Module):
    def __init__(self, model_config,
                 input_ch=3, input_ch_views=3):
        super(HashNeRF, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # move geo feat dim to one level higher?
        self.geo_feat_dim = model_config["sigma"]["geo_feat_dim"]

        self.make_sigma_net(model_config["sigma"])
        self.make_color_net(model_config["color"])
    
    def make_sigma_net(self, config):
        """
        create the sigma network
        (input: pos/xyz)
        """
        layers = config["layers"]
        hdim = config["hdim"]
        sigma_net = []
        for l in range(layers):
            if l == 0:
                sigma_net.append(nn.Linear(self.input_ch, hdim, bias=False))
            elif l == layers - 1:
                sigma_net.append(nn.Linear(hdim, 1 + self.geo_feat_dim, bias=False))
            else:
                sigma_net.append(nn.Linear(hdim, hdim, bias=False))
            sigma_net.append(nn.ReLU(True))
        self.sigma_net = nn.ModuleList(sigma_net[:-1]) # exclude the last relu

    def make_color_net(self, config):
        """
        create the color network
        (inputs: geo_feat from sigma net, viewdirs)
        """
        layers = config["layers"]
        hdim = config["hdim"]
        
        color_net = []
        for l in range(layers):
            if l == 0:
                color_net.append(nn.Linear(self.input_ch_views + self.geo_feat_dim, hdim, bias=False))
            elif l == layers - 1:
                color_net.append(nn.Linear(hdim, 3, bias=False))
            else:
                color_net.append(nn.Linear(hdim, hdim, bias=False))
            color_net.append(nn.ReLU(True))
        self.color_net = nn.ModuleList(color_net[:-1]) # exclude the last relu
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # (net_chunk, 3)

        # sigma
        h = input_pts
        for l in range(len(self.sigma_net)):
            h = self.sigma_net[l](h)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        # feed SH (geo_Feat) to color network
        # (net_chunk, 1),
        # (net_chunk, 15)
        
        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(len(self.color_net)):
            h = self.color_net[l](h)
            
        # (net_chunk, 3)
        # color = torch.sigmoid(h)
        # h is color
        outputs = torch.cat([h, sigma.unsqueeze(dim=-1)], -1)
        # (net_chunk, 3 + 1)

        return outputs