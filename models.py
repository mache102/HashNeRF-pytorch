"""
NeRF and NeRFSmall models

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List
from dataclasses import dataclass, field

@dataclass
class SigmaNetConfig:
    input_ch: int = 3
    layers: int = 3
    hdim: int = 64
    geo_feat_dim: int = 15
    skips: List[int] = field(default_factory=lambda: [4])

@dataclass
class ColorNetConfig:
    input_ch: int = 3
    layers: int = 4
    hdim: int = 64

class CustomizableNeRF(nn.Module):
    def __init__(self, typ,
                 sigma: SigmaNetConfig,
                 color: ColorNetConfig,
                 encode_appearance: bool = False, input_ch_a: int = 48,
                 encode_transient: bool = False, input_ch_t: int = 16,
                 beta_min: float = 0.03):
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
        """
        super().__init__()
        self.typ = typ
        self.sigma = sigma
        self.color = color

        """
        NerfW parameters
        """
        self.encode_appearance = encode_appearance
        self.input_ch_a = input_ch_a if encode_appearance else 0

        self.encode_transient = False if typ=='coarse' else encode_transient
        self.input_ch_t = input_ch_t

        self.beta_min = beta_min

        self.make_sigma_net(sigma)
        self.make_color_net(color)

        if self.encode_transient:
            self.make_transient_net(input_ch=input_ch_t, layers=4,
                                    hdim=sigma.hdim)

    def make_sigma_net(self, s):
        """
        Make MLP network that outputs sigma (density)
        and the inputs to the color network

        (MLP1 in nerfw paper)

        s = sigma
        """
        sigma_net = []
        for l in range(s.layers):
            if l == 0:
                layer = nn.Linear(s.input_ch, s.hdim)
            elif l in s.skips: # skips for sigma only - not present in color net
                layer = nn.Linear(s.hdim + s.input_ch, s.hdim)
            else:
                layer = nn.Linear(s.hdim, s.hdim)

            layer = nn.Sequential(layer, nn.ReLU(True))
            sigma_net.append(layer)
        
        self.sigma_net = nn.ModuleList(sigma_net)
        self.sigma_net_final = nn.Linear(s.hdim, s.hdim)   

        self.static_sigma = nn.Sequential(nn.Linear(s.hdim, 1), nn.Softplus())

    def make_color_net(self, c):
        """
        Make MLP network that outputs color (rgb)
        takes sigma_net output, viewdirs, and appearance embedding as input

        (MLP2 in nerfw paper)

        c = color
        """

        simple_net = False
        if simple_net:
            color_in = c.hdim + c.input_ch + self.input_ch_a
            self.color_net = \
                nn.Sequential(
                    nn.Linear(color_in, c.hdim // 2), 
                    nn.ReLU(True)
                )
        else:
            color_net = []

            for l in range(c.layers):
                # the final layer is set at static_color
                if l == 0:
                    layer = nn.Linear(c.input_ch + c.geo_feat_dim, c.hdim)
                else:
                    layer = nn.Linear(c.hdim, c.hdim)

                layer = nn.Sequential(layer, nn.ReLU(True))
                color_net.append(layer)
            
            self.color_net = nn.ModuleList(color_net)

        self.static_color = nn.Sequential(nn.Linear(c.hdim // 2, 3), nn.Sigmoid())

    def make_transient_net(self, input_ch: int, layers: int, hdim: int):
        """
        Make MLP network that outputs transient components
        (uncertainty: beta, color: rgb, density: sigma)

        (MLP3 in nerfw paper)
        """
        hdim2 = hdim // 2
                
        transient_net = []
        for l in range(layers):
            if l == 0:
                transient_net.append(nn.Linear(hdim + input_ch, hdim2))
            else:
                transient_net.append(nn.Linear(hdim2, hdim2))
            transient_net.append(nn.ReLU(True))

        self.transient_net = nn.Sequential(*transient_net)

        self.transient_sigma = nn.Sequential(nn.Linear(hdim2, 1), nn.Softplus())
        self.transient_color = nn.Sequential(nn.Linear(hdim2, 3), nn.Sigmoid())
        self.transient_beta = nn.Sequential(nn.Linear(hdim2, 1), nn.Softplus())

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
                torch.split(x, [self.sigma.input_ch,
                                self.color.input_ch + self.input_ch_a,
                                self.input_ch_t], dim=-1)
        else:
            input_sigma, input_color_a = \
                torch.split(x, [self.sigma.input_ch,
                                self.color.input_ch + self.input_ch_a], dim=-1)
            
        """
        sigma network

        input_sigma: xyz/pos inputs
        """
        sigma_ = input_sigma
        for i in range(self.sigma.layers):
            if i in self.sigma.skips:
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

        # concatenate signa encoded & (viewdirs + appearance embedding)
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
    

class VanillaNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


        
class NeRFGradient(NeRF):
    def __init__(self, **kwargs):
        """ 
        """
        super(NeRFGradient, self).__init__(**kwargs)
        if self.use_viewdirs:
            self.gradient_linear = nn.Linear(self.W//2, 3)
        
    def forward(self, x):
        if x.shape[-1] > self.input_ch + self.input_ch_views:
            input_pts, input_views, input_depth = torch.split(x, [self.input_ch, self.input_ch_views, 1], dim=-1)
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            gradient = self.gradient_linear(h)
            outputs = torch.cat([rgb, alpha, gradient], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

# Small NeRF for Hash embeddings
class NeRFSmall(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,
                 ):
        super(NeRFSmall, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        # (net_chunk, 3)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        print(h.shape)

        sigma, geo_feat = h[..., 0], h[..., 1:]
        # feed SH (geo_Feat) to color network
        # (net_chunk, 1),
        # (net_chunk, 15)
        
        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
            
        # (net_chunk, 3)
        # color = torch.sigmoid(h)
        color = h
        outputs = torch.cat([color, sigma.unsqueeze(dim=-1)], -1)
        # (net_chunk, 3 + 1)

        return outputs
    

if __name__ == '__main__':
    model = NeRFSmall(num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=3, input_ch_views=3,)
    
    x = torch.rand(10, 6)
    y = model(x)

    import pdb; pdb.set_trace()