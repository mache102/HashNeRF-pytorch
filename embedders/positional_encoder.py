import torch
import torch.nn as nn

"""
Positional encoding (section 5.1)
"""
class PosEmbedder(nn.Module):
    def __init__(self, max_log2, N_freqs, log_sampling=True, include_input=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if log_sampling:
            self.freqs = 2**torch.linspace(0, max_log2, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_log2, N_freqs)

        self.include_input = include_input
        self.out_dim = 6 * N_freqs + (3 if include_input else 0) # 3 for xyz

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        if self.include_input:
            out = [x]
        else: 
            out = []
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)