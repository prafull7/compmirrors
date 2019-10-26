import torch
import torch.nn as nn
from functools import partial
import itertools
import numpy as np

# XXX The global device id handling might be not very optimal here, should use those context prefixes...

# Basic upsampling layers for two- and three- dimensional tensors. Somewhat hardcoded assumptions but works fine for this purpose.
class Upsample2(nn.Module):
    def __init__(self, mode='nearest'):
        super(Upsample2, self).__init__()
        self.mode = mode

    def forward(self, x):
        xs = x.shape
        #x = nn.functional.interpolate(x, scale_factor=2, align_corners=True, mode=self.mode)
        x = nn.functional.interpolate(x, scale_factor=2, mode=self.mode)
        return x
        
class Upsample3(nn.Module):
    # there's a split mechanism here whereby we could scale a part of the variables with one method and rest with other, but it's unused for now (and maybe broken)
    def __init__(self, mode='nearest', split=0):
        super(Upsample3, self).__init__()
        self.mode = mode
        self.split = split

    def forward(self, x):

        xs = x.shape

        # why can't they all be called just 'linear'...
        linearnames = {3: 'linear', 4: 'bilinear', 5: 'trilinear'}
        mode = self.mode
        if mode == 'linear':
            mode = linearnames[len(x.shape)]


        if self.split > 0:
            [x0, x1] = torch.split(x, [self.split, xs[1]-self.split], 1)
            x0 = nn.functional.interpolate(x0, scale_factor=2, align_corners=True, mode=linearnames[len(x.shape)])
            if self.mode == 'linear':
                x1 = nn.functional.interpolate(x1, scale_factor=2, align_corners=True, mode=mode)
            else:
                x1 = nn.functional.interpolate(x1, scale_factor=2, mode=mode)
            #x = nn.functional.interpolate(x, scale_factor=2, mode=self.mode)
            x = torch.cat((x0,x1), 1)
        else:
            if self.mode == 'linear':
                x = nn.functional.interpolate(x, scale_factor=2, align_corners=True, mode=mode)     # ... also why did they make it illegal to supply align_corners if it's not used by the method
            else:
                x = nn.functional.interpolate(x, scale_factor=2, mode=mode)


        return x


# Also a couple of packaged-up convolution layers, one used in TNet and other in VNet.
# These contain some extra functionality, like the coordinate feature maps etc., and some unused (and possibly broken) features too.
# For development-historical reasons, they are unnecessarily complicated/duplicate, and there are some less-than-logical differences between them, so read carefully if
# details are important. Could be unified into one properly configurable convolution layer, but they do the job.
class TConv(nn.Module):
    def __init__(self, sizes,
                 activation=partial(nn.functional.leaky_relu, negative_slope=0.1),
                 fsize=(3,3), auxfeats=None, coords=True, padfunc=nn.ReplicationPad2d,
                 noise=False, window=False, linear_window=False):
        super(TConv, self).__init__()
        self.activation = activation
        self.sizes = sizes
        self.fsize = fsize

        self.coords = coords
        self.coords_dim = 2 if coords else 0
        self.coords_cached = False

        self.noise = noise
        self.noise_dim = 4 if noise else 0
        self.noise_cached = False


        self.window = window
        self.linear_window = linear_window
        self.window_cached = False

        self.pad = padfunc((fsize[0]-fsize[0]//2-1,fsize[0]//2,fsize[1]-fsize[1]//2-1,fsize[1]//2))


        self.convs = nn.ModuleList()
        for s_in, s_out in zip(self.sizes, self.sizes[1:]):
            self.convs.append(nn.Conv2d(s_in + self.coords_dim + self.noise_dim, s_out, self.fsize, bias=True))


        for conv in self.convs:
            nn.init.xavier_normal(conv.weight)
            #conv.bias.data.fill_(0.000)
            nn.init.normal(conv.bias, std=0.001)


    def forward(self, x):
        xs = x.shape

        
        if self.coords:
            # If this is the first time this layer is evaluated, generate the linear gradient coordinate feature maps.
            if not self.coords_cached:
                self.ci = torch.linspace(-1, 1, xs[2], device=device).cuda().view(1,1,xs[2],1).expand(xs[0], -1, -1, xs[3])
                self.cj = torch.linspace(-1, 1, xs[3], device=device).cuda().view(1,1,1,xs[3]).expand(xs[0], -1, xs[2], -1)
                self.coords_cached = True
            # Thereafter just cat these cached maps onto the features
            x = torch.cat((x,self.ci,self.cj), dim=1)
            xs = x.shape

        if self.noise:
            # We could insert (fixed) random noise feature maps, but disabled in current version
            if not self.noise_cached:
                self.N = torch.randn((1,self.noise_dim,xs[2],xs[3]), device=device).cuda().expand((xs[0],-1,-1,-1))
                self.noise_cached = True
            x = torch.cat((x,self.N), dim=1)
            xs = x.shape

        for conv in self.convs:
            x = self.pad(x)
            x = conv(x)


        if self.activation is not None:
            x = self.activation(x)

        if self.window:
            if not self.window_cached:
                # Similar caching mechanism as with coords above.
                # The linear_window version is not used.
                if not self.linear_window:
                    W = np.matmul(
                        np.reshape(np.hanning(xs[2]), [xs[2],1]),
                        np.reshape(np.hanning(xs[3]), [1,xs[3]]))
                    W = np.reshape(W, [1,1,xs[2],xs[3]])
                    W = np.sqrt(W)
                    self.W = torch.from_numpy(W).float().to(device)
                else:
                    Wi = torch.linspace(0, 2, xs[2], device=device).cuda().view(1,1,xs[2],1).expand(xs[0], xs[1]//4, -1, xs[3])
                    Wj = torch.linspace(0, 2, xs[3], device=device).cuda().view(1,1,1,xs[3]).expand(xs[0], xs[1]//4, xs[2], -1)
                    self.W = torch.cat((Wi,Wj,1-Wi,1-Wj), dim=1)


                self.window_cached = True

            x = x * self.W


        return x



class VConv(nn.Module):
    def __init__(self, sizes,
                 activation=partial(nn.functional.leaky_relu, negative_slope=0.1), fsize=(3,3,3), auxfeats=None,
                 coords=True, padfunc=nn.ReplicationPad3d,
                 noise=False, window=False):
        super(VConv, self).__init__()
        self.activation = activation
        self.sizes = sizes
        self.fsize = fsize

        self.coords = coords
        self.coords_dim = 3 if coords else 0
        self.coords_cached = False

        self.noise = noise
        self.noise_dim = 3 if noise else 0
        self.noise_cached = False

        self.window = window
        self.window_cached = False

        self.pad = padfunc((fsize[0]-fsize[0]//2-1, fsize[0]//2,
                            fsize[1]-fsize[1]//2-1, fsize[1]//2,
                            fsize[2]-fsize[2]//2-1, fsize[2]//2))

        self.convs = nn.ModuleList()
        for s_in, s_out in zip(self.sizes, self.sizes[1:]):
            self.convs.append(nn.Conv3d(s_in + self.coords_dim + 1*self.noise_dim, s_out, self.fsize, bias=True))


        for conv in self.convs:
            nn.init.xavier_normal(conv.weight)
            nn.init.normal(conv.bias, std=0.2)


    def forward(self, x):
        xs = x.shape

        if self.coords:
            if not self.coords_cached:
                self.ci = torch.linspace(-1, 1, xs[2], device=device).cuda().view(1,1,xs[2],1,1).expand(xs[0], -1, -1, xs[3],xs[4])
                self.cj = torch.linspace(-1, 1, xs[3], device=device).cuda().view(1,1,1,xs[3],1).expand(xs[0], -1, xs[2], -1,xs[4])
                self.ck = torch.linspace(-1, 1, xs[3], device=device).cuda().view(1,1,1,1,xs[4]).expand(xs[0], -1, xs[2], xs[3],-1)
                self.coords_cached = True
            x = torch.cat((x,self.ci,self.cj,self.ck), dim=1)
            xs = x.shape

        if self.noise:
            if not self.noise_cached:
                self.N = torch.randn((1,self.noise_dim,xs[2],xs[3],xs[4]), device=device).cuda().expand((xs[0],-1,-1,-1,-1))
                self.noise_cached = True
            x = torch.cat((x, self.N), dim=1)
            xs = x.shape

        for conv in self.convs:
            x = self.pad(x)
            x = conv(x)


        if self.window:
            if not self.window_cached:
                W = np.matmul(
                    np.reshape(np.hanning(xs[3]), [xs[3],1]),
                    np.reshape(np.hanning(xs[4]), [1,xs[4]]))
                W = np.reshape(W, [1,1,1,xs[3],xs[4]])
                W = np.sqrt(W)
                self.W = torch.from_numpy(W).float().to(device)
                self.window_cached = True

            if 1:
                x = x * self.W
            else:
                xmean = torch.mean(x, dim=(3,4), keepdim=True)
                x = x * self.W + xmean * (1-self.W)
        if self.activation is not None:
            x = self.activation(x)

        return x


