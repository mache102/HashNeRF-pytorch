import torch
import torch.nn as nn

class VanillaNeRF(nn.Module):
    def __init__(self, model_config, input_chs,
                 use_viewdirs=False, use_gradient=False):
        """ 
        Vanilla NeRF with an option to use gradient (OmniNeRF).
        """
        super(VanillaNeRF, self).__init__()
        layers = model_config['layers']
        hdim = model_config['hdim']
        output_ch = model_config['output_ch']
        self.skips = model_config['skips']
        input_chs = input_chs
        
        self.use_viewdirs = use_viewdirs
        self.use_gradient = use_gradient

        self.make_sigma_net(layers, hdim, input_chs["xyz"])
        self.make_color_net(layers, hdim, input_chs["dir"])

        if use_viewdirs:
            self.feature_linear = nn.Linear(hdim, hdim)
            self.alpha_linear = nn.Linear(hdim, 1)
            self.rgb_linear = nn.Linear(hdim // 2, 3)
            if self.use_gradient:
                self.gradient_linear = nn.Linear(hdim // 2, 3)
        else:
            self.output_linear = nn.Linear(hdim, output_ch)

    def make_sigma_net(self, layers, hdim, input_ch):   
        sigma_net = []
        for l in range(layers - 1):
            if l == 0:
                sigma_net.append(nn.Linear(input_ch, hdim))
            elif l in self.skips:
                sigma_net.append(nn.Linear(hdim + input_ch, hdim))
            else:
                sigma_net.append(nn.Linear(hdim, hdim))
            sigma_net.append(nn.ReLU(True)) # relu
        self.sigma_net = nn.ModuleList(sigma_net)

    def make_color_net(self, layers, hdim, input_ch, use_code_release=True):
        if use_code_release:
            ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
            self.color_net = nn.ModuleList([nn.Linear(input_ch + hdim, hdim // 2)])

        else:
            ### Implementation according to the paper
            color_net = []
            for l in range(layers // 2):
                if l == 0:
                    color_net.append(nn.Linear(input_ch + hdim, hdim // 2))
                else:
                    color_net.append(nn.Linear(hdim // 2, hdim // 2))
                color_net.append(nn.ReLU(True)) # relu
            self.color_net = nn.ModuleList(color_net)

    def forward(self, x):
        # gradient option
        if x.shape[-1] > self.input_ch + self.input_ch_views:
            # input depth is unused(?)
            input_pts, input_views, input_depth = torch.split(x, [self.input_ch, self.input_ch_views, 1], dim=-1)
        else:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = input_pts
        for i in range(len(self.sigma_net)):
            h = self.sigma_net[i](h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i in range(len(self.color_net)):
                h = self.color_net[i](h)

            rgb = self.rgb_linear(h)
            if self.use_gradient:
                gradient = self.gradient_linear(h)
                outputs = torch.cat([rgb, alpha, gradient], -1)
            else:
                outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs  

# 2023 1019: merged nerfgradient with vanillanerf  