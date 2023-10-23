"""
NeRF and NeRFSmall models

"""
import torch
import torch.nn as nn

class NeRFW(nn.Module):
    def __init__(self, model_config, input_chs,
                 use_appearance: bool = False, 
                 use_transient: bool = False):
        """
        Modified from https://github.com/kwea123/nerf_pl/blob/nerfw/models/nerf.py

        ---Parameters---
        model_config: config for the model (coarse or fine)
        input_chs: input channels (xyz, viewdirs, appearance, transient)
        use_appearance: whether to use appearance encoding
        use_transient: whether to use transient encoding

        ---Model Config---    
        "sigma": {
            "layers": Number of layers for density (sigma) net
            "hdim": Number of hidden units in each layer
            "geo_feat_dim": Dimension of the geometry feature (to be concatenated with viewdirs)
        },
        "color": {
            "layers": Number of layers for color net
            "hdim": Number of hidden units in each layer
        },
        "appearance": {
            "input_ch": Number of input channels for appearance
        },
        "transient": {
            "input_ch": Number of input channels for transient
            "layers": Number of layers for transient net
            "hdim": Number of hidden units in each layer
        },
        "beta_min": Minimum pixel color variance

        do not encode transient for coarse model
        """
        super().__init__()

        self.input_chs = input_chs 
        if not use_appearance: 
            self.input_chs["appearance"] = 0

        self.sigma_skips = model_config["sigma"]["skips"]
        self.beta_min = model_config["beta_min"]

        self.make_sigma_net(model_config["sigma"])
        self.make_color_net(model_config["color"])

        if use_transient:
            self.make_transient_net(model_config["transient"])

    def make_sigma_net(self, s):
        """
        Make MLP network that outputs sigma (density)
        and the inputs to the color network

        (MLP1 in nerfw paper)

        s = sigma
        """
        hdim = s["hdim"]
        layers = s["layers"]

        sigma_net = []
        for l in range(layers):
            if l == 0:
                layer = nn.Linear(self.input_chs["xyz"], hdim)
            elif l in s.skips: # skips for sigma only - not present in color net
                layer = nn.Linear(hdim + self.input_chs["xyz"], hdim)
            else:
                layer = nn.Linear(hdim, hdim)

            layer = nn.Sequential(layer, nn.ReLU(True))
            sigma_net.append(layer)
        
        self.sigma_net = nn.ModuleList(sigma_net)
        self.sigma_net_final = nn.Linear(hdim, hdim)   

        self.static_sigma = nn.Sequential(nn.Linear(hdim, 1), nn.Softplus())

    def make_color_net(self, c):
        """
        Make MLP network that outputs color (rgb)
        takes sigma_net output, viewdirs, and appearance embedding as input

        (MLP2 in nerfw paper)

        c = color
        """
        hdim = c["hdim"]
        layers = c["layers"]
        geo_feat_dim = c["geo_feat_dim"]

        simple_net = False
        if simple_net:
            color_in = hdim + self.input_chs["dir"] + self.input_chs["appearance"]
            self.color_net = \
                nn.Sequential(
                    nn.Linear(color_in, hdim // 2), 
                    nn.ReLU(True)
                )
        else:
            color_net = []

            for l in range(layers):
                # the final layer is set at static_color
                if l == 0:
                    layer = nn.Linear(self.input_chs["dir"] + geo_feat_dim, hdim)
                else:
                    layer = nn.Linear(hdim, hdim)

                color_net.extend((layer, nn.ReLU(True)))
            
            self.color_net = nn.Sequential(*color_net)

        self.static_color = nn.Sequential(nn.Linear(c.hdim // 2, 3), nn.Sigmoid())

    def make_transient_net(self, t):
        """
        Make MLP network that outputs transient components
        (uncertainty: beta, color: rgb, density: sigma)

        (MLP3 in nerfw paper)
        """
        hdim = t["hdim"]
        layers = t["layers"]

        transient_net = []
        for l in range(layers):
            if l == 0:
                transient_net.append(nn.Linear(hdim + self.input_chs["transient"], hdim // 2))
            else:
                transient_net.append(nn.Linear(hdim // 2, hdim // 2))
            transient_net.append(nn.ReLU(True))

        self.transient_net = nn.Sequential(*transient_net)

        self.transient_sigma = nn.Sequential(nn.Linear(hdim // 2, 1), nn.Softplus())
        self.transient_color = nn.Sequential(nn.Linear(hdim // 2, 3), nn.Sigmoid())
        self.transient_beta = nn.Sequential(nn.Linear(hdim // 2, 1), nn.Softplus())

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        Encodes input (sigma+color) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (color + viewdirs + appearance + transient)
            sigma_only: whether to infer sigma only.
            output_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_ony:
                static_sigma
            elif output_transient:
                static_color, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_color, static_sigma

        assume full (output transient is true)

        sigma -> mlp1 -> density (static_sigma out)
        """
        # unpack inputs
        # NOTE: input_color_a contains both viewdirs and appearance embeddings
        if sigma_only:
            input_sigma = x
        elif output_transient:
            input_sigma, input_color_a, input_t = \
                torch.split(x, [self.input_chs["xyz"],
                                self.input_chs["dir"] + self.input_chs["appearance"],
                                self.input_chs["transient"]], dim=-1)
        else:
            input_sigma, input_color_a = \
                torch.split(x, [self.input_chs["xyz"],
                                self.input_chs["dir"] + self.input_chs["appearance"]], dim=-1)
            
        """
        sigma network

        input_sigma: xyz/pos inputs
        """
        sigma_ = input_sigma
        for i in range(len(self.sigma_net)):
            if i in self.sigma_skips:
                sigma_ = torch.cat([input_sigma, sigma_], 1)
        
            sigma_ = self.sigma_net[i](sigma_)

        # obtain sigma (static) outputs
        static_sigma = self.static_sigma(sigma_) # (B, 1)
        if sigma_only:
            return static_sigma

        # prepare sigma encoding for transient net & static color net
        sigma_net_final = self.sigma_net_final(sigma_)

        """
        color network
        sigma_net_final: output from the sigma net
        input_color_a: viewdirs & appearance embeddings, concatenated

        # concatenate sigma encoded & (viewdirs + appearance embedding)
        # input to color net 
        # finally obtain color (static) outputs
        # which concludes the static outputs
        """
        color_net_input = torch.cat([sigma_net_final, input_color_a], 1)
        color_ = self.color_net(color_net_input)
        static_color = self.static_color(color_) # (B, 3)
        static = torch.cat([static_color, static_sigma], 1) # (B, 4)
        if not output_transient:
            return static

        """
        transient network

        sigma_net_final: output from the sigma net
        input_t: transient input
        """
        transient_input = torch.cat([sigma_net_final, input_t], 1)
        transient_ = self.transient_net(transient_input)

        # separate calculation for transient outputs
        transient_sigma = self.transient_sigma(transient_) # (B, 1)
        transient_rgb = self.transient_rgb(transient_) # (B, 3)
        transient_beta = self.transient_beta(transient_) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)

if __name__ == '__main__':
    pass 