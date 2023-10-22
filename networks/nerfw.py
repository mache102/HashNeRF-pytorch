"""
NeRF and NeRFSmall models

"""
import torch
import torch.nn as nn

class NeRFW(nn.Module):
    def __init__(self, model_config,
                 input_ch, input_ch_views,
                 input_ch_a, input_ch_t,
                 appearance: bool = False, 
                 transient: bool = False):
        """
        https://github.com/kwea123/nerf_pl/blob/nerfw/models/nerf.py

        ---Parameters for the original NeRF---
        layers_sigma: number of layers for density (sigma) net
        hdim: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        input_ch_sigma: number of input channels for sigma (3+3*10*2=63 by default)
        input_ch_color: number of input channels for colorection (3+3*4*2=27 by default)
        input_ch_t: number of input channels for t

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        input_ch_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        input_ch_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance


        do not encode transient for coarse model
        """
        super().__init__()

        self.input_ch = input_ch 
        self.input_ch_views = input_ch_views
        self.input_ch_a = input_ch_a if appearance else 0
        self.input_ch_t = input_ch_t

        self.sigma_skips = model_config["sigma"]["skips"]
        self.beta_min = model_config["beta_min"]

        self.make_sigma_net(model_config["sigma"])
        self.make_color_net(model_config["color"])

        if transient:
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
                layer = nn.Linear(self.input_ch, hdim)
            elif l in s.skips: # skips for sigma only - not present in color net
                layer = nn.Linear(hdim + self.input_ch, hdim)
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
            color_in = hdim + self.input_ch_views + self.input_ch_a
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
                    layer = nn.Linear(self.input_ch_views + geo_feat_dim, hdim)
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
                transient_net.append(nn.Linear(hdim + self.input_ch_t, hdim // 2))
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
                torch.split(x, [self.input_ch,
                                self.input_ch_views + self.input_ch_a,
                                self.input_ch_t], dim=-1)
        else:
            input_sigma, input_color_a = \
                torch.split(x, [self.input_ch,
                                self.input_ch_views + self.input_ch_a], dim=-1)
            
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
    model = HashNeRF(num_layers=3,
                 hdim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,)
    
    x = torch.rand(10, 6)
    y = model(x)

    import pdb; pdb.set_trace()