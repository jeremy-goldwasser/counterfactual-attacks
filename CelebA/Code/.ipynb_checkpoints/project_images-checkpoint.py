import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image
import torchvision
import torchvision.transforms as transforms

import pickle
import pathlib


# gan_path = /PATH/TO/GAN
gan_path = os.path.expanduser("~/Counterfactuals/stylegan3-projector/results/network-snapshot-008800.pkl")
# celebaDir = /PATH/TO/CELEBA/DATA
celebaDir = os.path.expanduser("~/Counterfactuals/celeba/img_align_celeba")

import subprocess
git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
projectionDir = os.path.join(git_root, "CelebA", "Data")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
with open(gan_path, 'rb') as fp:
    G = pickle.load(fp)['G_ema'].to(device)

zs = torch.randn([100000, G.mapping.z_dim], device=device)
w_stds = G.mapping(zs, None).std(0)


#@markdown #**Run the inversion** 🚀
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision.transforms.functional as TF
import time
# from tqdm.notebook import tqdm
from tqdm import tqdm
from dnnlib.util import open_url

url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)

def get_target_features(target):
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    return vgg16(target_images, resize_images=False, return_lpips=True)

def get_perceptual_loss(synth_image, target_features):
    # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    synth_image = (synth_image + 1) * (255/2)
    if synth_image.shape[2] > 256:
        synth_image = F.interpolate(synth_image, size=(256, 256), mode='area')

    # Features for synth images.
    synth_features = vgg16(synth_image, resize_images=False, return_lpips=True)
    return (target_features - synth_features).square().sum()


def project_image(projection_target, 
                  steps = 1000, 
                  plot_freq = 200,
                  lr = 0.1,
                  save_images=False,
                  seed = None):
    tf = Compose([
      Resize(224),
      lambda x: torch.clamp((x+1)/2,min=0,max=1),
    ])
    if seed is not None:
        torch.manual_seed(seed)
    target_features = get_target_features(projection_target)

    # Initialize a single 512-dimensional vector
    # q_single = torch.randn([1, G.mapping.z_dim], device=device).requires_grad_()
    with torch.no_grad():
        qs = []
        losses = []
        for _ in range(32):
            q = (G.mapping(torch.randn([4,G.mapping.z_dim], device=device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
            w = q * w_stds + G.mapping.w_avg
            images = G.synthesis(w)
            loss = get_perceptual_loss(images, target_features)
            i = torch.argmin(loss)
            qs.append(q[i])
            losses.append(loss)
        qs = torch.stack(qs)
        losses = torch.stack(losses)
        i = torch.argmin(losses)
        q_single = qs[i][0].requires_grad_()


    # stylegan2 projector uses these params with basic Adam
    opt = torch.optim.AdamW([q_single], lr=lr, betas=(0.9, 0.999))

    # loop = tqdm(range(steps))
    loop = range(steps)
    for i in loop:
        opt.zero_grad()

        # Repeat the vector 16 times to form a 16x512 matrix
        q = q_single.repeat(16, 1)  

        # Compute style vectors 'w'
        w = (q * w_stds + G.mapping.w_avg)

        # Synthesize image using the expanded vector
        image_recon = G.synthesis(w.unsqueeze(0), noise_mode='const')
        loss = get_perceptual_loss(image_recon, target_features)
        loss.backward()
        opt.step()

        ct = 0 if i==0 else i+1
        if plot_freq is not None:
            if ct % plot_freq == 0:
                print(f"image {ct}/{steps} | loss: {loss.item()}")
                display(TF.to_pil_image(tf(image_recon)[0]))

    pil_image = TF.to_pil_image(image_recon[0].add(1).div(2).clamp(0, 1))

    timestring = time.strftime('%Y%m%d%H%M%S')
    os.makedirs(f'samples/{timestring}', exist_ok=True)
    if save_images:
        pil_image.save(f'samples/{timestring}/{i:04}.jpg')
    w_vec = w[0]
    return w_vec


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop((178, 178)),
    torchvision.transforms.Resize((256, 256))
])

dataset = torchvision.datasets.ImageFolder(root=celebaDir, transform=transform)

NUM_IMS = 10000
w_vecs = []
for im_idx in tqdm(range(NUM_IMS), desc="Processing images"):
    if im_idx == 0 or (im_idx + 1) % 100 == 0:
        print(f"{im_idx + 1}")

    tensor_og = dataset[im_idx][0].to(device)
    projection_target = tensor_og * 255.0

    w_vec = project_image(
        projection_target,
        steps=500,
        plot_freq=None,
        lr=0.1,
        save_images=False,
        seed=None
    )

    w_vecs.append(w_vec.cpu())

# Stack and save the final result
w_fname_all = os.path.join(projectionDir, 'w_all_10k.pt')
X = torch.stack(w_vecs)
torch.save(X, w_fname_all)