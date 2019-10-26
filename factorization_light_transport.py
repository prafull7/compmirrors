import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import numpy as np
import math

import scipy.misc
from scipy.misc import imsave
import os

from functools import partial
import itertools

import data_loading
import video_svd
import layers
import model_T
import model_V

from data_loading import *
from video_svd import *

from layers import *
from model_T import *
from model_V import *

import visdom
import argparse
import datetime
import random
import socket

machine_name = socket.gethostname()

parser = argparse.ArgumentParser(description='Parse arguments for the code.')

parser.add_argument('-d', '--data_dir', type=str,
    default=os.environ.get('FACTORIZE_DATA_DIR', None), help='Location for data directory')
parser.add_argument('-o', '--out_dir', type=str,
    default=os.environ.get('FACTORIZE_OUT_DIR', None), help='Location for output directory')
parser.add_argument('-f', '--folder', type=str,
    default='last_final', help='Name of the folder')
parser.add_argument('-ds', '--dataset', type=str,
    default='lastfinal', help='Name of the dataset')
parser.add_argument('-s', '--seq_name', type=str,
    default='hands', help='Name of the sequence')
parser.add_argument('-n_ds', '--num_downsample', type=int,
    default=3, help='Num downsamples')
parser.add_argument('-dev', '--device', type=int,
    default=0, help='GPU ID')
parser.add_argument('-sf', '--skip_frames', type=int,
    default=0, help='How many frames to skip (fast forward)')
parser.add_argument('-lr', '--lr', type=float,
    default=0.00006, help='Learning rate')
parser.add_argument('-miters', '--max_iters', type=int,
    default=100000, help='Maximum iterations')
parser.add_argument('-vis_a', '--visdom_address', type=str,
    default='localhost', help='Network address of the visdom server')
parser.add_argument('-vis_p', '--visdom_port', type=int,
    default=8097, help='Port of the visdom server')
parser.add_argument('-vis_v', '--visdom_video', type=int,
    default=1, help='Output the estimate of the hidden video to visdom. Requires ffmpeg command line.')

parser.add_argument('-nvec', '--num_singular_vectors', type=int,
    default=32, help='Number of singular vectors')
parser.add_argument('-vdim', '--latent_video_dim', type=int,
    default=16, help='Dimension of the latent video (e.g. 16 for a 16x16 pixel resolution)')
parser.add_argument('-color', '--use_color', type=int,
    default=1, help='Use RGB color (1) or grayscale (0) for both latent video and transport matrix')

args = parser.parse_args()
cur_time = datetime.datetime.now()

visdom_url = args.visdom_address
visdom_port = args.visdom_port
visdom_video = args.visdom_video

visdom_env = str(cur_time.strftime("%Y-%m-%d-%H%M--"))+ args.seq_name + "_" + args.dataset + "_" + socket.gethostname() + "_" + str(args.device)
if args.skip_frames != 0:
    visdom_env = visdom_env + "ff"

vis = visdom.Visdom(visdom_url, port=visdom_port, env=visdom_env)

device = args.device
print("Device ID:", device)
torch.cuda.set_device(device)

data_dir = args.data_dir
folder_name = args.folder
dataset = args.dataset
seq_name = args.seq_name

skip_frames = args.skip_frames

layers.device = device
model_T.device = device
model_V.device = device
frames_dict = json.load(open(os.path.join(args.data_dir, args.folder, "frames.txt"),'r'))
data_loading.frames_dict = frames_dict

# for visdom
tvmean_record = []
loss_record = []
loss_all_record = []
tracks_all_record = []


def tensor_to_viz(x):
    return np.flipud(np.squeeze(x.detach().cpu().numpy()))


# a little helper class for record keeping and outputting in a consistent way
class FLoss():
    def __init__(self, tensor, name='loss', weight=1):
        self.weight = weight
        self.tensor = tensor
        self.name = name

    def eval(self):
        if self.weight == 0:
            return 0
        else:
            return self.weight * self.tensor

    def item(self):
        if self.weight == 0:
            return 0
        else:
            return (self.weight * self.tensor).item()


# The main program
def factorize():
    created_outputdir = False
    
    nvec = args.num_singular_vectors
    v_size = args.latent_video_dim
    color = args.use_color

    Z, overexp_mask, gt_imgs = load_observations(data_dir, folder_name, dataset, seq_name, skip_frames, n_downsample=args.num_downsample)
    # Z contains the observed video in shape [t i j c] (time, latent video shape height, widh, color channels)

    if not color:
        # if monochrome, create a dummy color channel of size 1
        # (not sure if the remainder of the code still works properly with monochrome, there might be some hardcoded assumptions of 3 channels but probably easy to fix)
        Z = Z[:,:,:,1]
        Z = np.expand_dims(Z,3)

    # for convenient reference later
    s_i = v_size        # latent video height
    s_j = v_size        # ... width
    s_I = Z.shape[1]    # observed video height
    s_J = Z.shape[2]    # ... width
    s_c = Z.shape[3]    # color channels in either video
    s_b = 1             # batch size (no foreseeable reason why this would be other than 1 but let's keep it around)
    s_t = Z.shape[0]    # number of observed frames

    # Pre-compute the SVD's, separately for all three color channels.
    # Get the SVD for the 0th channel first (this is a bit clumsy)
    sv_U, sv_S, sv_V = video_svdbasis(Z[:,:,:,0], k=nvec)
    sv_U = np.expand_dims(sv_U, 0)
    sv_S = np.expand_dims(sv_S, 0)
    sv_V = np.expand_dims(sv_V, 0)
    # if there are more channels, concatenate
    for c in range(Z.shape[3]-1):
        sv_Uc, sv_Sc, sv_Vc = video_svdbasis(Z[:,:,:,c+1], k=nvec)
        sv_U = np.concatenate((sv_U, np.expand_dims(sv_Uc, 0)), 0)
        sv_S = np.concatenate((sv_S, np.expand_dims(sv_Sc, 0)), 0)
        sv_V = np.concatenate((sv_V, np.expand_dims(sv_Vc, 0)), 0)

    # A bit of additional processing to multiply in the half-power of the singular values and to shape into TNet's assumed format
    # In a notational deviation from the paper, we happen to be working with transposes of the matrices so it's the V vectors we want.
    sv_V_aux = sv_V[:,:nvec,:]
    sv_V_aux = np.reshape(sv_V_aux, [s_c, nvec, s_I, s_J])  # [c, sv, I, J]
    sv_V_aux = sv_V_aux * np.reshape(np.sqrt(sv_S/np.expand_dims(sv_S[:,1],1)), [s_c, nvec, 1, 1])
    svecs = torch.from_numpy(sv_V_aux).float().to(device)

    sv_V = torch.from_numpy(sv_V).float().to(device)    # [sv, IJ]
    sv_S = torch.from_numpy(sv_S).float().to(device)
    sv_U = torch.from_numpy(sv_U).float().to(device)    # [t, sv]

    Z = torch.from_numpy(Z).float().to(device)
    print("Observed video shape: ", Z.shape)

    # A mask that excludes pixels which had overexposure in any frame of Z.
    # As clipping is a nonlinear operation, it would break the linearity assumptions underlying the method.
    # However there is no problem if we just ignoring those pixels, other than loss of potentially useful info.
    oemask = 1.0-overexp_mask.astype(np.float32)
    # Also for excluding finite differences in smoothness priors:
    oemask_dI = oemask[1:,:] * oemask[:-1,:]
    oemask_dJ = oemask[:,1:] * oemask[:,:-1]

    oemask = torch.from_numpy(oemask).float().to(device)  
    oemask_dI = torch.from_numpy(oemask_dI).float().to(device)  
    oemask_dJ = torch.from_numpy(oemask_dJ).float().to(device)  

    Zmean = torch.mean(Z, dim=0)    # average observed frame
    Zmeanmean = torch.mean(Zmean)   # average intensity of the entire observed video
    

    # A template for the two optimizers
    optimizer_partial = partial(optim.Adam, amsgrad=True)

    # Build the prediction networks.
    # The notation is a bit different from the paper. Here TNet refers to \mathcal{Q} (the singular vector weight prediction network) 
    # and VNet to \mathcal{L} (the hidden video prediction network)
    tnet = TNet_SV(svecs=svecs, Zmean=Zmean, out_channels=s_c)
    tnet = tnet.to(device)
    tnet_optimizer = optimizer_partial(list(tnet.parameters()), lr=args.lr) 

    vnet = VNet(s_t, out_channels=s_c)
    vnet = vnet.to(device)
    vnet_optimizer = optimizer_partial(list(vnet.parameters()), lr=args.lr)


    criterion1 = nn.L1Loss()
    criterion = nn.MSELoss()


    # A pre-loss tonemapping function. In principle it could be a good idea to use something like the log below, but it's disabled for now.
    # Note that non-linearity would be fine here, as it's just for the loss, after all the linear ops are done.
    def tonemap(x):
        #return torch.clamp((x+0.01), min=0.01).log()
        return x

    # NOTE: we're overwriting Z with its tonemapped version here, careful later
    Z = tonemap(Z)
    ZT = Z.permute(3, 1, 2, 0).view(1, s_c, s_I, s_J, s_t)

    ZT_di = ZT[:, :, 1:, :, :] - ZT[:, :, :-1, :, :]    # finite difference in vertical direction
    ZT_dj = ZT[:, :, :, 1:, :] - ZT[:, :, :, :-1, :]    # finite difference in horizontal direction


    for iter in range(args.max_iters):
        # Visualize things in visdom every 30 frames; let's set a flag here if the data should be prepared
        vis_images = True if iter % 30 == 0 else False

        # Just some random visualization parameters
        rf = np.random.randint(0, s_t-1 - 8) 
        ri = v_size//2
        rj = v_size//2

        # Here we'll build a list of losses, both for final addition as the optimization loss, and for viz
        losses = []
        loss = 0

        print('iter %i' % (iter))
        tnet_optimizer.zero_grad()
        vnet_optimizer.zero_grad()

        v = vnet()      # this corresponds to L in the paper. It comes out in shape [1 c t i j]
        t, A = tnet()   # t in shape [1 c I J i j]; we're also getting the A-matrix (in paper it's Q)

        # Get the shapes for easy reference
        vs = v.shape
        ts = t.shape

        # Collect some data for visualization later, if needed
        if vis_images:
            vsnap_full = nn.functional.interpolate(v.detach().cpu().squeeze(0), scale_factor=8, mode='nearest').numpy()
            tsnap_full = t.detach().cpu().numpy()

        # TODO: clean up
        vsnap = v[0,0,rf,:,:].squeeze()
        tsnap = t[0, 0, ri, rj, :, :].squeeze()
        tsnap3 = t[0,0,20,87,:,:].squeeze()
        vsnap2 = v[0,0,:,ri,:].squeeze()
        tsnap2 = t[0,0,s_I//5,:,ri,:].squeeze()

        # reshape t and v into something we can matrix-multiply
        # Minor note: a careful reader might have observed that one could bypass building T explicitly above, and insted go directly to T*L by arranging the multiplications
        # of the singular vectors differently. Might be more efficient in some cases. However then it's harder to e.g. implement the nonnegativity prior, but there are ways around
        # this (e.g. building only a random subset of T slices explicitly, and making that loss stochastic).
        # Also there exists an fairly simple explicit formula for the loss ||Z-TL||_2^2 using the singular values only and no need to build the full matrices, which the reader can work
        # out if interested. In principle it's much much faster and less memory-hungry to evaluate, but limited to the L2. We're not using any of this here but it
        # could be of interest in the future.
        t = t.view(s_b,s_c,s_I*s_J,s_i*s_j)
        v = v.permute(0,1,3,4,2).view(s_b,s_c,s_i*s_j,s_t) / (s_i*s_j)  # note scaling by pixel count

        # Perform the actual matrix multiplication
        tv = torch.matmul(t,v)  # b c HW t

        tv = tv.view(s_b,s_c,s_I,s_J,s_t)   # b c H W t
        tvmean = torch.mean(tv) # debug

        # Apply the tonemapping to the predicted video, so that it's comparable to the observed video
        tv = tonemap(tv)

        # Loss computation
        oemask_ = oemask.view(1,1,s_I,s_J,1)
        loss_fit_direct = 0.01*criterion(oemask_*tv,oemask_*ZT)     # an L2 loss on the pixel-wise similarity between the predicted and observed videos, with overexposed pixels discarded 
        losses.append(FLoss(loss_fit_direct, 'fit_direct'))

        roff = np.random.randint(1, 8)  # use a random temporal difference between 1 to 8 frames (averages stochastically to a uniform distribution of steps)
        tv_dt = tv[:,:,:,:,roff:]-tv[:,:,:,:,:-roff]    # temporal difference of prediction
        ZT_dt = ZT[:,:,:,:,roff:]-ZT[:,:,:,:,:-roff]    # ... and observation
        loss_fit_dt = criterion1(oemask_*tv_dt, oemask_*ZT_dt)  # L1 loss on their difference, again with overexposed pixels ignored
        losses.append(FLoss(loss_fit_dt, 'fit_dt'))


        # the magnitude loss, to anchor the values a bit
        A = tnet.A.view(s_c, nvec, s_i, s_j)
        loss_mag = 0.0001 * torch.mean(A[:,0,:,:].abs())
        losses.append(FLoss(loss_mag, 'mag'))

        smooth = 0.001
        Zmean_tm = tonemap(Zmean).view(s_b, s_c, s_I, s_J, 1).expand(-1, -1, -1, -1, s_i * s_j)
        t_tm = tonemap(t).view(s_b, s_c, s_I, s_J, s_i*s_j)
        if smooth > 0:
            t_tm2 = t_tm.view(s_b, s_c, s_I, s_J, s_i, s_j)
            # Smoothness loss on predicted T, on finite differences along both I and J
            # (on further thought the overexposure mask wouldn't be needed here, but let's keep it for consistency with the paper results)
            loss_smooth_dI = smooth * torch.mean(oemask_dI.view(1,1,s_I-1,s_J,1,1) * torch.abs(t_tm2[:,:,1:,:,:,:]-t_tm2[:,:,:-1,:,:,:]))
            losses.append(FLoss(loss_smooth_dI, 'smooth_dI'))
            loss_smooth_dJ = smooth * torch.mean(oemask_dJ.view(1,1,s_I,s_J-1,1,1) * torch.abs(t_tm2[:,:,:,1:,:,:]-t_tm2[:,:,:,:-1,:,:]))
            losses.append(FLoss(loss_smooth_dJ, 'smooth_dJ'))

        # Nonnegativity loss on T: discourage individual pixels that are below 0 by a large penalty
        loss_nonneg = 10*torch.mean(torch.pow(torch.clamp(t,max=0),2))
        losses.append(FLoss(loss_nonneg, 'nonneg'))

        # chromaticity loss: penalize the deviation of the T colors from their average
        t_Y = torch.mean(t, dim=1, keepdim=True)
        loss_T_chroma = 0.001*criterion1(t,t_Y)
        losses.append(FLoss(loss_T_chroma, 'T_chroma'))


        loss += torch.sum(torch.stack([loss.eval() for loss in losses]))
        loss.backward()
        tnet_optimizer.step()
        vnet_optimizer.step()

        if iter % 10 == 0:
            # update the loss plots every 10 frames
            loss_all = [loss.item() for loss in losses]
            loss_legend = [loss.name for loss in losses]
            loss_all_record.append(loss_all)
            vis.line(X=np.array(np.arange(len(loss_all_record))), Y=np.log10(1e-7+np.array(loss_all_record)), win='loss_breakdown', 
                opts = {
                    'title': 'Loss breakdown (log)',
                    'xlabel': 'iters (x10)', 
                    'ylabel': 'Loss', 
                    'legend': loss_legend,
                })


            tvmean_record.append(tvmean.item())
            loss_record.append(loss.item())

        # Save at every 1000 iterations
        if (iter) % 1000 == 0:
            print("Saving at iter: ", iter+1)
            if not created_outputdir:
                # Create output folder with a hash and commit the current code to track it.
                created_outputdir = True
                hash_code = random.getrandbits(32)
                print(args.out_dir, machine_name, args.seq_name, args.dataset)
                outdir = args.out_dir + '/run_' + str(cur_time.strftime("%Y-%m-%d-%H%M--")) + "_" + machine_name + "_" + args.seq_name + "_" + args.dataset 
                if args.skip_frames != 0:
                    outdir = outdir + "ff"
                outdir = outdir + '/'
                try:
                    os.mkdir(outdir)
                except OSError:
                    # TODO: was this an actual error condition, or just "directory already exists" catch?
                    None
                
            np.save(outdir + 'v.npy', v.detach().cpu().numpy())
            np.save(outdir + 't.npy', t.detach().cpu().numpy())
            torch.save(tnet, outdir + 'tnet.pth')
            torch.save(vnet, outdir + 'vnet.pth')

            def to8bit(x):
                return np.uint8(np.maximum(0.0, np.minimum(1.0, x)) * 255.)

            outdir_iter = os.path.join(outdir, 'vid_%08i' % iter)
            vsnap_full /= np.max(vsnap_full)
            vsnap_full = np.power(vsnap_full, 0.4545)   # gamma correction

            try:
                os.mkdir(outdir_iter)
            except OSError:
                None
                
            print("before saving check: ", s_t, np.min(gt_imgs), np.max(gt_imgs), np.min(np.transpose(np.squeeze(vsnap_full[:,0,:,:]), (1,2,0))), np.max(np.transpose(np.squeeze(vsnap_full[:,0,:,:]), (1,2,0)))) 
            print(np.hstack((to8bit(np.transpose(np.squeeze(vsnap_full[:,0,:,:]), (1,2,0))), gt_imgs[0])).shape)
                  
            for f in range(s_t):
                img_gen = np.hstack((to8bit(np.transpose(np.squeeze(vsnap_full[:,f,:,:]), (1,2,0))), gt_imgs[f]))
                img = img_gen
                imsave(os.path.join(outdir_iter, 'f_%04d.png' % f), img)
            
            # tsnap [1 c I J i j]
            tsnap_full /= np.max(tsnap_full)
            tsnap_full = np.power(tsnap_full, 0.4545)   # gamma correction
            c = 0
            for i in range(s_i):
                for j in range(s_j):
                    #vsnap_full -= np.min(vsnap_full)
                    img_gen = np.transpose(np.squeeze(tsnap_full[0,:,:,:,i,j]), (1,2,0))
                    #img_gt = np.squeeze(Z[f,:,:].detach().cpu().numpy())
                    #img = np.concatenate((img_gen,img_gt),1)
                    img = img_gen
                    imsave(os.path.join(outdir_iter, 't_%06d.png' % c), to8bit(img))
                    c += 1
            for j in range(s_j):
                for i in range(s_i):
                    #vsnap_full -= np.min(vsnap_full)
                    img_gen = np.transpose(np.squeeze(tsnap_full[0,:,:,:,i,j]), (1,2,0))
                    #img_gt = np.squeeze(Z[f,:,:].detach().cpu().numpy())
                    #img = np.concatenate((img_gen,img_gt),1)
                    img = img_gen
                    imsave(os.path.join(outdir_iter, 't_%06d.png' % c), to8bit(img))
                    c += 1

            if visdom_video:
                os.system("ffmpeg -i " + os.path.join(outdir_iter, 'f_%04d.png ') +  os.path.join(outdir_iter, 'f_vid.mp4\n'))
                vis.video(videofile=os.path.join(outdir_iter, 'f_vid.mp4'), opts=dict(autoplay=True, loop=True), win="f_video")
                os.system("ffmpeg -i " + os.path.join(outdir_iter, 't_%06d.png ') +  os.path.join(outdir_iter, 't_vid.mp4\n'))
                vis.video(videofile=os.path.join(outdir_iter, 't_vid.mp4'), opts=dict(autoplay=True, loop=True), win="t_video")

        
        # This happens at every 30 iterations by default.
        # Just dump a big pile of potentially interesting debug visualizations.
        # These are not necessarily all very logically chosen or relevant but they were used to get some insight and intuition during development, so let's just leave them here.
        if vis_images:

            #v = v.permute(0,2,3,4,1).view(s_b,s_c,s_i*s_j,s_t) / (s_i*s_j)
            v_vis = v.view([s_b,s_c,s_i,s_j,s_t]) / torch.max(v)  # b c i j t
            v_vis_nsqrt = 4
            v_vis_n = v_vis_nsqrt ** 2
            v_vis_frames = np.floor(np.linspace(0,s_t-1, 16))
            v_vis_imgs = v_vis[0,:,:,:,v_vis_frames]
            v_vis_imgs = v_vis_imgs.permute(3,0,1,2)    # id c i j
            v_vis_imgs = nn.functional.interpolate(torch.sqrt(v_vis_imgs), scale_factor=4, mode='nearest')
            vis.images(v_vis_imgs, v_vis_nsqrt, opts = {'caption':'video frames'}, win='vframes')

            A_vis_nsqrt = np.floor(np.sqrt(nvec)).astype(np.int)
            A_vis_n = A_vis_nsqrt ** 2
            A_vis = tnet.A[0,0,:A_vis_n,:].view(A_vis_n,1,v_size*v_size) 
            A_vis = A_vis - torch.mean(A_vis, dim=2, keepdim=True)
            A_vis = torch.clamp(A_vis / (0.0000001+torch.std(A_vis,dim=2,keepdim=True)) * 0.13 + 0.5, min=0, max=1)
            A_vis = A_vis.view(A_vis_n,1,v_size,v_size) 
            A_vis = nn.functional.interpolate(A_vis, scale_factor=4, mode='nearest') * 255
            vis.images(A_vis, A_vis_nsqrt, opts = {'caption':'A slices'}, win='aslices')


            vis.heatmap(tensor_to_viz(tv[0,0,:,:,rf]), opts=dict(colormap='Hot', title="Computed TV"),  win='V11')
            vis.heatmap(tensor_to_viz(ZT[0,0,:,:,rf]), opts=dict(colormap='Hot', title="Z"), win='V12')
            vis.heatmap(tensor_to_viz(tv[0,0,:,:,rf] - ZT[0,0,:,:,rf]), opts=dict(colormap='Hot', title="Z - TV"), win='qminust')
            vis.heatmap(tensor_to_viz(vsnap), opts=dict(colormap='Hot', title="vsnap"), win='V13')
            vis.heatmap(tensor_to_viz(tsnap), opts=dict(colormap='Hot', title="tsnap"), win='V21')
            vis.heatmap(np.transpose(tensor_to_viz(vsnap2)), opts=dict(colormap='Hot', title="vsnap2"), win='V22')
            vis.heatmap(tensor_to_viz(tsnap), opts=dict(colormap='Hot', title="tsnap"), win='V23')

            
            A = tnet.A[0,0,:,:].view(nvec,v_size,v_size).detach().cpu().numpy()
            s = np.random.randint(0,15)
            vis.heatmap(np.flipud(np.squeeze(A[s,:,:])), opts=dict(colormap='Hot', title="T Slice A[s,:,:]"), win='V31')
            vis.heatmap(np.flipud(np.squeeze(A[:,s,:])), opts=dict(colormap='Hot', title="T Slice A[:,s,:]"), win='V32')

            t = t.view(s_b,s_c,s_I,s_J,s_i,s_j)
            Tslices1 = torch.cat((
                t[0,0,:,s_J//2,:,s_j//2].squeeze(),
                t[0,0,:,s_J//2,s_i//2,:].squeeze(), 
                ), dim=1).t()
            
            Tslices2 = torch.cat((
                t[0,0,s_I//2,:,s_i//2,:].squeeze(), 
                t[0,0,s_I//2,:,:,s_j//2].squeeze(),
                ), dim=1).t()

            vis.heatmap(tensor_to_viz(Tslices1), opts=dict(colormap='Hot', title="T slices vert"), win='Tslices vertical')
            vis.heatmap(tensor_to_viz(Tslices2), opts=dict(colormap='Hot', title="T slices horz"), win='Tslices horizontal')



factorize()
