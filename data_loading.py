import torch
import cv2
import numpy as np
from utils import *
import pickle
import glob
import os
import json
import re

# This is a natural sort function for list of strings taken from 
# https://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# Returns filenames for a given path.
def get_filenames(path, ext):
    files = natural_sort(glob.glob(path + "*"))
    files = [f for f in files if f.endswith(ext)]
    return files

# This function takes the sequence of images, and an integer skip_frames
# and averages every skip_frames number of frames to reduce the number of 
# output frames to len(images)/skip_frames. It also chops off the tail of the
# resulting sequence to ensure the length of output sequence is multiple of 8.
def avg_skip_frames(images, skip_frames):
    N, H, W, C = images.shape
    print(images.shape)
    
    # Making number of frames multiple of skip_frames
    if skip_frames != 0:
        N -= N % skip_frames
        images = images[0:N]
        images = images.reshape(int(len(images)/skip_frames), skip_frames, H, W, C)
        images = np.squeeze(np.mean(images, axis=1))

    # Make number of frames a multiple of 8
    N, H, W, C = images.shape
    N -= N % 8
    images = images[0:N]
    return images

# Loads pgm files in the given path in index [file_start, file_end]. 
def load_pgm(path, file_start, file_end, num_downsample, ext=".pgm"):
    images = []
    files = get_filenames(path, ext)
    overexp_mask = None
    for i in range(file_start, file_end):
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
        img = img[0:img.shape[1]-img.shape[1]%(2**num_downsample), 0:img.shape[1]-img.shape[1]%(2**num_downsample)]
        overexp = (img > (2**16 - 100)).astype(np.float32)

#         if np.max(np.array(img)) > 2**16 - 1000:
#             print('WARNING: frame %i, overexposed pixels %i' % (i, np.sum(img > 2**16-1000)))

        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        colored = np.array(img, dtype=np.float32)

        for _ in range(num_downsample):
            colored = downsample_box_half_mono(colored)
            overexp = downsample_box_half_mono(overexp)
        images.append(colored)
        if overexp_mask is not None:
            overexp_mask += overexp
        else:
            overexp_mask = overexp

    overexp_mask = overexp_mask > 0    
    return np.array(images), overexp_mask

# This function loads the ground truth frames.
def load_gt(path, file_start, file_end, ext=""):
    images = []
    print("Loading GT: \n Start index: ", file_start, "\t End index:", file_end)
    files = get_filenames(path, '')
    for i in range(file_start, file_end):
        img = cv2.imread(files[i], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        img = (img - np.min(img))/(np.max(img) - np.min(img)) * 255
        if len(img.shape) == 2:
            print("Img size before: ", img.shape)
            img = np.stack([img for i in range(3)], axis=2)
            print("Img size:", img.shape)
        images.append(img)
    images = np.array(images)
    return images

# Skips frames if skip_frames is turned on. This is for running long sequences which 
# we ran in fast-forward mode by averaging every skip_frames, making the video fast forward.
def skip_frames_gt(images, skip_frames):
    N, H, W, C = images.shape
    
    # Make number of frames multiple of skip_frames
    if skip_frames != 0:
        N -= N % skip_frames
        images = images[0:N]
        images = images.reshape(int(len(images)/skip_frames), skip_frames, H, W, C)
        images = np.squeeze(images[:, 0, :, :, :])

    # Make number of frames a multiple of 8
    N, H, W, C = images.shape
    N -= N % 8
    images = images[0:N]
    return images
    
# This function reads the observed frames (Z matrix in Z = T*L) from the data_dir.
# In our data, all observed frames are marked with substring reflector.
# It also downsamples the video spatially and also zeros out overexposed pixels.
def load_observations(data_dir, folder_name, dataset, seq_name, skip_frames=0, n_downsample=3, zero_overexp=True):
    blacks = None
    blocks_oe = None
    
    # Read black frames if available
    if 'black' in frames_dict: 
        blacks, blacks_oe = load_pgm(os.path.join(data_dir, folder_name, dataset + "_reflector"), frames_dict['black'][0], frames_dict['black'][1], n_downsample)

    imgs, imgs_oe = load_pgm(os.path.join(data_dir, folder_name, dataset + "_reflector"),
                         frames_dict[seq_name][0], frames_dict[seq_name][1], n_downsample)
    
    gt_imgs = load_gt(os.path.join(data_dir, folder_name, dataset + "_target"), frames_dict[seq_name][0], frames_dict[seq_name][1], ext="")
   
    imgs = avg_skip_frames(imgs, skip_frames)
    gt_imgs = skip_frames_gt(gt_imgs, skip_frames)

    overexp = imgs > (2**16-50)
    overexp = np.any(overexp, axis=0)

    imgs = imgs / 2**14
    
    # Subtracting the ambient light term by subtracting the average black 
    # frame (if available) from each frame of the observed video.
    if blacks is not None:
        blacks = blacks / 2**14
        blacks = np.mean(blacks, axis=0, keepdims=True)
        imgs = imgs - blacks

    imgs[:,0,0:32] = imgs[:,1,0:32]
    if zero_overexp:
        imgs = imgs * (1.0-np.expand_dims(np.expand_dims(imgs_oe,2),0).astype(np.float32))

    Z = imgs
    Z[:,0,0:2] = Z[:,1,0:2]  # Replacing the blinking pixel at corner of all frames, this is encoding some camera information related to time.
    return Z, imgs_oe, gt_imgs

