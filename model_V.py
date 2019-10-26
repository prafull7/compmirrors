import torch
import torch.nn as nn
from functools import partial
import itertools
from layers import *
import numpy as np


# video prediction network
class VNet(nn.Module):

    def __init__(self, nframes, out_channels=1, dark=True):
        super(VNet, self).__init__()

        self.nframes = nframes
        self.out_channels = out_channels

        self.dark = dark    # use of mixed tanh-exp-activation -- if true, two images will be predicted, and combined as seen in forward()

        # random noise input seed 2x2 mini-video with one-eighth the number of frames of the full video
        self.S0 = [4, nframes//8, 2, 2]
        self.x0 = torch.nn.Parameter(torch.randn([1] + self.S0, requires_grad=True, device=device))

        # learnable "background color level"
        self.blacklevel = torch.nn.Parameter(torch.zeros([1,out_channels,1,1,1], dtype=torch.float, requires_grad=True, device=device))


        layerwidth = 64

        # Note, coords are true by default for VConv
        self.layers_pre = \
            nn.Sequential(
                VConv([self.S0[0], layerwidth], window=False),
                Upsample3(),
                VConv([layerwidth, layerwidth], window=False),
                VConv([layerwidth, layerwidth]),
                Upsample3(),
                VConv([layerwidth, layerwidth]),
                VConv([layerwidth, layerwidth]),
                VConv([layerwidth, layerwidth], coords=False, noise=False, window=False),
                Upsample3(),
                VConv([layerwidth, layerwidth], coords=False, noise=False, window=False),
                VConv([layerwidth, layerwidth//2], coords=False, noise=False, window=False),
                VConv([layerwidth // 2, self.out_channels if not dark else self.out_channels*2], activation=None, coords=False, noise=False, window=False),
            )

    def forward(self):
        if self.dark:
            # run the network
            xr = self.layers_pre(self.x0)
            # apply the positivity etc. activations
            x = self.blacklevel.exp() * (0.5+0.5*torch.tanh(xr[:,0:self.out_channels,:,:,:]+2))
            x = x + (xr[:,self.out_channels:self.out_channels*2,:,:,:] - 1.0).exp()
        else:
            x = self.layers_pre(self.x0)
            x = (x-1.0).exp()
            x = x + self.blacklevel.exp()
        return x
