# This script implements the matrix factorization experiment in the paper Figure 3.
# Of course, every run will give a different result (and possibly a different degree
# of success) due to the random initialization. For the default input images, it
# often takes around 10k-20k iterations for the method to settle.
# 
# The script visualizes the result
# in a Matplotlib window that updates as the iteration proceeds. The input and 
# ground truth data are first displayed, and a keypress is expected for the
# computation to start.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import numpy as np
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import argparse
import scipy.misc

parser = argparse.ArgumentParser(description='Parse arguments for the code.')

parser.add_argument('-T', '--T_image', type=str,
    default='./data/inputs_1d/lightfield.png', help='T matrix image file')
parser.add_argument('-L', '--L_image', type=str,
    default='./data/inputs_1d/tracks_bg.png', help='L matrix image file')
parser.add_argument('-o', '--out_dir', type=str,
    default='output_1d', help='Location for output directory')
parser.add_argument('-i', '--iters', type=int,
    default=99999999, help='Number of iterations (by default run forever)')


# These correspond to notation in Figure 3 of the paper
parser.add_argument('-sh', '--sh', type=int,
    default=128, help='Rows of T matrix')
parser.add_argument('-sq', '--sq', type=int,
    default=128, help='Cols of T matrix / Rows of L matrix')
parser.add_argument('-sw', '--sw', type=int,
    default=256, help='Cols of T matrix')

parser.add_argument('-n', '--noiselevel', type=float,
    default=0.0, help='Noise level injected to the input (for stress testing -- try e.g. 0.1 with the default input images)')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# were w u t
h = args.sh
q = args.sq
w = args.sw


T0 = mpimg.imread(args.T_image)
L0 = mpimg.imread(args.L_image)

# extract the green channel if it's RGB and not mono
if len(T0.shape) == 3:
    T0 = T0[:,:,2]

if len(L0.shape) == 3:
    L0 = L0[:,:,2]

# Resize to the specified (compatible) dimensions, and scale intensities to range [0,1] (assuming these are 8-bit images)
T0 = scipy.misc.imresize(T0, [h, q]).astype(float) / 255
L0 = scipy.misc.imresize(L0, [q, w]).astype(float) / 255

# Form the input matrix
T0L0 = T0.dot(L0)
# Corrupt it noise, if any is specified in the command line arguments
T0L0 += np.random.normal(0, args.noiselevel, T0L0.shape)

# Visualize the starting situation and wait for key press
fig = plt.figure()
plt.clf()
plt.subplot(2, 2, 1).set_title('T ground truth')
plt.imshow(T0)
plt.subplot(2, 2, 2).set_title('L ground truth')
plt.imshow(L0)
plt.subplot(2, 2, 3).set_title('TL (input to factorization)')
plt.imshow(T0L0)
#plt.subplot(2, 2, 4).set_title('TL horizontal gradient')
#plt.imshow((T0L0[:, 2:-1] - T0L0[:, 1:-2]))
plt.subplot(2, 2, 4).set_title('press any key to start')

plt.waitforbuttonpress()

T0 = torch.Tensor(T0).to(device)
L0 = torch.Tensor(L0).to(device)
T0L0 = torch.Tensor(T0L0).to(device)


# factor that the images get upsampled by from the initial noise
dsfactor = 32

ldim = 64
latent_pad = 0  # we can pad the final image with some to-be-discarded pixels if desired, to reduce boundary effects, but not sure if it's any help (disabled)

T_inputsize = [T0.shape[0] // dsfactor + latent_pad, T0.shape[1] // dsfactor + latent_pad]
L_inputsize = [L0.shape[0] // dsfactor + latent_pad, L0.shape[1] // dsfactor + latent_pad]

# These are the noises fed to the network. They're set as learnable.
T_latent = torch.randn(1, ldim, T_inputsize[0], T_inputsize[1], requires_grad=True, device=device)
L_latent = torch.randn(1, ldim, L_inputsize[0], L_inputsize[1], requires_grad=True, device=device)

# Number and width of the layers is indicated here
layerwidths = [ldim, 32, 64, 64, 128, 128]

# We inject as auxiliary info the mean of the input matrix along dim 1 in some layers. Could do so for dim 0 too.
TLmean = T0L0.sum(dim=1, keepdim=True).repeat(1,T0.shape[1]).view(1,1,T0.shape[0],T0.shape[1])
TLmean -= TLmean.mean() # standardize
TLmean /= (TLmean*TLmean).mean().sqrt()

class FNet(nn.Module):
    def __init__(self, feats, apply_coords = True, inputsize = None, aux_channels = None):
        super(FNet, self).__init__()

        coordpad = 0
        if apply_coords:
            coordpad += 2

        self.aux_channels = aux_channels

        self.aux_channels_pad = self.aux_channels.shape[1] if self.aux_channels is not None else 0

        self.convs = nn.ModuleList([nn.Conv2d(feats[i] + coordpad, feats[i + 1], 4) for i in range(0, len(feats) - 1)])

        for conv in self.convs:
            nn.init.xavier_normal(conv.weight)
            conv.bias.data.fill_(0.001)

        # A bit messy but does the job
        self.conv_final1 = nn.Conv2d(feats[-1] + coordpad + self.aux_channels_pad, feats[-1]//2, 3)
        nn.init.xavier_normal(self.conv_final1.weight)
        self.conv_final1.bias.data.fill_(0.0)
        self.conv_final2 = nn.Conv2d(feats[-1]//2 + coordpad, 1, 3)
        nn.init.xavier_normal(self.conv_final2.weight)
        self.conv_final2.bias.data.fill_(0.0)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_last = nn.Upsample(scale_factor=2, mode='bilinear')
        self.drop = nn.Dropout2d(p=0.5)
        self.pad = nn.ReplicationPad2d((1,2,1,2))
        self.pad_final = nn.ReplicationPad2d((1,1,1,1))

    # build the "coordconv" style coordinate images along the image dimensions, to be concatenated as aux feature maps
    def makecoords(self, shape):
        a = torch.linspace(-1, 1, shape[2])
        b = torch.linspace(-1, 1, shape[3])
        cx = a.cuda().view(shape[2], 1).repeat(1, shape[3]).view(1, 1, shape[2], shape[3])
        cy = b.cuda().repeat(shape[2]).view(1, 1, shape[2], shape[3])
        return cx, cy


    def forward(self, x):
        x = self.drop(x)
        for i in range(0, len(self.convs)):
            x = self.pad(x)
            cx, cy = self.makecoords(x.shape)
            x = torch.cat((x,cx,cy), 1)
            x = self.convs[i](x)

            if i == len(self.convs) - 1:
                x = self.up_last(x)
            else:
                x = self.up(x)

            x = F.tanh(x)

        cx,cy = self.makecoords(x.shape)
        x = torch.cat((x, cx, cy), 1)
        if self.aux_channels is not None:
            x = torch.cat((x, self.aux_channels), 1)
        x = self.pad_final(x)
        x = self.conv_final1(x)
        x = F.leaky_relu(x, 0.1)    # this is a slightly odd choice after all the tanh's, but it was here when we ran the experiments so let's keep it for reproducibility...

        cx,cy = self.makecoords(x.shape)
        x = torch.cat((x, cx, cy), 1)
        x = self.pad_final(x)
        x = self.conv_final2(x)

        x = torch.exp(x-2)  # -2 just to bring down the numerical range a bit, not significant
        return x

def centercrop(x, tw, th):
    xs = x.shape
    x = x.view(xs[2], xs[3])
    x1 = int(round((xs[2] - tw) / 2.))
    y1 = int(round((xs[3] - th) / 2.))
    return x[x1:x1 + tw, y1:y1 + th]

# Note that we're using the same FNet class for both T and L, but it's a different instance. No need to make separate classes in this case, because we want an identical architecture.
tnet = FNet(layerwidths, inputsize=T_inputsize, aux_channels=TLmean)
tnet = tnet.cuda()

lnet = FNet(layerwidths, inputsize=L_inputsize)
lnet = lnet.cuda()

# just run one batch to get the sizes
T = tnet(T_latent)
Ts = T.size()

optimizer = optim.Adam(list(tnet.parameters()) + list(lnet.parameters()) + [T_latent, L_latent], lr=0.0001, weight_decay=0.0001)
criterion = nn.L1Loss()

try:
    os.mkdir(args.out_dir)
except OSError:
    # TODO: was this an actual error condition, or just "directory already exists" catch?
    None


for iter in range(args.iters):

    optimizer.zero_grad()

    # predict the matrices
    T = tnet(T_latent)
    L = lnet(L_latent)

    # if there was extra padding use, discard the boundary regions
    T = centercrop(T, T0.shape[0], T0.shape[1])
    L = centercrop(L, L0.shape[0], L0.shape[1])

    # get the matrix product of the predicted matrices
    TL = torch.mm(T,L)

    # direct pointwise loss (with very small weight) and gradient losses along i and j
    loss = 0.0001*criterion(TL, T0L0)
    loss += criterion(TL[:,2:-1]-TL[:,1:-2], T0L0[:,2:-1]-T0L0[:,1:-2])
    loss += criterion(TL[2:-1:,]-TL[1:-2:,], T0L0[2:-1,:]-T0L0[1:-2,:])

    loss.backward()
    optimizer.step()

    if iter % 10 == 0:
        print('iter %i \tloss: %f' % (iter, loss.item()))

    # display and save the factor images at every 100 iters
    if iter % 100 == 0:
        plt.clf()
        plt.subplot(2,2,1).set_title('T')
        plt.imshow(T.detach().cpu().numpy())
        
        plt.subplot(2,2,2).set_title('L')
        plt.imshow(L.detach().cpu().numpy())
        plt.subplot(2,2,3).set_title('TL')
        plt.imshow(TL.detach().cpu().numpy())
        # the horizontal gradient is slightly interesting for the "lightfield" experiment, but not a fundamentally important component of anything
        plt.subplot(2,2,4).set_title('horiz. gradient of TL')
        plt.imshow((TL[:, 2:-1] - TL[:, 1:-2]).detach().cpu().numpy())
        fig.canvas.draw()
        fig.canvas.flush_events()

        Limg = L.detach().cpu().numpy()
        Limg = Limg / np.amax(Limg)
        mpimg.imsave('%s/l_%08d.png' % (args.out_dir, iter), Limg)

        Timg = T.detach().cpu().numpy()
        Timg = Timg / np.amax(Timg)
        mpimg.imsave('%s/t_%08d.png' % (args.out_dir, iter), Timg)
        
        # Obviously if one were to use the result somewhere, we would save the actual matrices too


print("done")
plt.waitforbuttonpress()
