"""
Exposure network, urban radiance fields

the third component is an exposure compensation network
that takes as an input an exposure latent code and estimates
the affine transformation to be applied to the color values
output by the radiance field network. There is a different
affine transformation per image. This compensates for the
different exposures across input images. 


Input: exposure latent code. supervised by the network itself
output: (3, 3) matrix, affine transformation to be applied to the color values
as well as a (3, 1) vector, which is added to the color values.

nerfw latent code: dim=48
urf: dim=4 (best)

so, basically: 
exposure input dim=4. output should be 3x3 matrix and 3x1 vector
vec is only described as part of the diagram in the appendix so I'm not sure about it 
"""
import torch 
import torch.nn as nn

class ExposureNet(nn.Module):
    def __init__(self, dim=4):
        super(ExposureNet, self).__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )

    def forward(self, x):
        # x: (bsz, dim)
        # output: (bsz, 3, 3)
        x = self.fc(x)
        x = x.view(-1, 3, 3)
        return x