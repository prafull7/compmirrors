# This file implements the network that predicts the transport matrix T

import torch 
import torch.nn as nn
import numpy as np
from functools import partial
import itertools
from layers import *


# transport matrix prediction network (singular-vector-based)
class TNet_SV(nn.Module):

    def __init__(self, svecs, Zmean, out_channels=1, neural=True):
        super(TNet_SV, self).__init__()

        self.neural = neural
        self.out_channels = out_channels
        self.S0 = [32,2,2]  # 32 feature channels in 2x2 seed "image"

        # the input to the network (trainable, so let's just start with zeros (should maybe be randn but probably not broken))
        self.x0 = torch.nn.Parameter(torch.zeros([1] + self.S0, requires_grad=True, device=device))

        self.svs = svecs.permute(0,2,3,1)   # c, I, J, sv  (TODO, clean up unnecessary permutations etc.)
        self.s_c = self.svs.shape[0]
        self.s_I = self.svs.shape[1]
        self.s_J = self.svs.shape[2]
        self.nvec = self.svs.shape[3]
        self.svs = self.svs.view(1,self.s_c, self.s_I*self.s_J, self.nvec)  # ready format for channel-batched matmul

        self.Zmean = Zmean.permute(2,0,1).view(1,self.s_c,self.s_I,self.s_J,1,1)


        if neural:
            layerwidth = 64
            self.svweight = torch.nn.Parameter(
                torch.tensor(np.ones([1,self.out_channels,self.svs.shape[3],1]).astype(np.float), dtype=torch.float,
                             requires_grad=True, device=device))


            # NOTE: here we are assuming a fixed number of upsamples, so this effectively sets the target resolution to 16x16!
            # If other resolutions are desired, the code needs to be changed (or ideally generalized a bit).
            self.layers = \
                nn.Sequential(
                    TConv([self.S0[0], layerwidth//2]),     # note currently coords=True by default
                    Upsample2(),
                    TConv([layerwidth//2, layerwidth]),
                    TConv([layerwidth, layerwidth]),
                    TConv([layerwidth, layerwidth]),
                    Upsample2(),
                    TConv([layerwidth, layerwidth], window=True),
                    TConv([layerwidth, layerwidth], window=True),
                    TConv([layerwidth, layerwidth], window=True),
                    Upsample2(),
                    TConv([layerwidth, layerwidth*2], coords=False),
                    TConv([layerwidth*2, layerwidth*4], coords=False),
                    TConv([layerwidth*4, self.out_channels * self.nvec], coords=False, activation=None),
                )

        else:
            # We aren't using this, but here's the optional way where we optimize directly over the entries of A instead of using the CNN.
            # It doesn't completely fail (depending on other parts), but is much less reliable.
            A0 = np.random.normal(0,1,[self.svs.shape[3], 16*16]).astype(np.float)
            A0[0, :] = 0
            A0 = A0 * 0.1

            self.A = torch.nn.Parameter(
                torch.tensor(A0, dtype=torch.float, requires_grad=True, device=device))


    def forward(self):
        if self.neural:
            # run the network
            x = self.layers(self.x0) 
            xs = x.shape
            # matrix multiply by a weighting based on the singular values
            x = x.view(xs[0],self.out_channels,self.nvec,xs[2]*xs[3])     # batch, c, s, ij
            x = x * self.svweight
            self.A = x  # just store and expose for viz
            # perform the actual expansion into full T using the singular vectors
            x = torch.matmul(self.svs, x)           # batch, c, I, J, ij
            xs = x.shape
            x = x.view(1,self.out_channels,self.s_I,self.s_J,16,16)  # XXX hardcoded assumption of 16x16, see also comment above at layer definitions
            # so now: [1, c, I, J, i, j]
            x = x + self.Zmean  # add the mean image by default, so the network only needs to care about generating the difference from mean (which is in span of SV's anyway)
        else:
            x = torch.matmul(self.svs, self.A)
            xs = x.shape
            x = x.view(xs[0],1,xs[1],xs[2],16,16).permute(0,1,4,5,2,3)
            x = x + self.Zmean

        return x, self.A

