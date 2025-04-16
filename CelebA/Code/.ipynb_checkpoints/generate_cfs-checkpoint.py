import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import os

cfDir = os.path.expanduser("~/Counterfactuals")
styleganDir = os.path.join(cfDir, 'stylegan3-projector')

# CelebA images and neural network predictors
# celebaDir = /PATH/TO/CELEBA/DATA
celebaDir = os.path.join(cfDir, 'celeba')
imDir = os.path.join(celebaDir, 'img_align_celeba') # Images themselves
predictorDir = os.path.join(celebaDir, 'predictors')

import subprocess
git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
outputDir = os.path.join(git_root, "CelebA", "Data")

import sys
sys.path.append(styleganDir)

import torch_utils
import dnnlib
import pickle

from SimpleCNNarchitecture import *
import projection
import helper
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

to_pil = transforms.ToPILImage()

# Evaluate model performance on subset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((178, 178)),
    transforms.Resize((256, 256))
])


# Load GAN
gan_path = os.path.join(styleganDir, 'results', 'network-snapshot-008800.pkl')
with open(gan_path, 'rb') as fp:
    G = pickle.load(fp).to(device)

# Load latent vectors and order
with open(os.path.join(outputDir, 'w_order.txt'), "r") as f:
    w_fnames = [line.strip() for line in f]
w_fname_all = os.path.join(outputDir, 'w_all_10k.pt')
X = torch.load(w_fname_all).detach().numpy()

# Prepare to run
Labels = ['Young', 'Attractive', 'Male', 'Smiling']
n_for_cf = 100
upper_thresh = 0.75; lower_thresh = 0.25
data_rows = []

for Label in Labels:
    weights_path = os.path.join(predictorDir, Label.lower() + '_weights.pt')
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    n_cfs = 0
    w_im_idx = 0
    while n_cfs < n_for_cf:
        print(f"{Label} index {w_im_idx}")
        w_fname = w_fnames[w_im_idx]
        
        # Load image
        idx_fname = w_fname.split('.')[0].split('_')[1]
        num_fname = str(int(idx_fname)+1)
        num_zeros_needed = 6 - len(num_fname)
        padded_num_fname = '0' * num_zeros_needed + num_fname
        jpg_fname = padded_num_fname + '.jpg'
        image = Image.open(os.path.join(imDir, jpg_fname))
        
        # Reconstructed image
        w_vec = torch.tensor(X[int(idx_fname),]).to(device)
        w = torch.stack((w_vec,)*G.num_ws)
        
        with torch.no_grad():
            image_og_recon = helper.latent_to_image(G, w).clamp(0, 1)[0]
            pred_recon = torch.sigmoid(model(image_og_recon)).item()
            
        if pred_recon < 0.5:
            w_cf_vec = helper.generate_counterfactual(G, w_vec, model, threshold=upper_thresh, 
                                                      binary=True, learning_rate=0.01, verbose=False)
        else:
            w_cf_vec = helper.generate_counterfactual(G, w_vec, model, threshold=lower_thresh, 
                                                      binary=True, direction='decreasing', learning_rate=0.01, verbose=False)
        if w_cf_vec is None:
            print("Zero gradient, skipping.")
            w_im_idx += 1
            continue
        w_cf = torch.stack((w_cf_vec,)*G.num_ws)
        
        w_cf_sklearn = w_cf_vec.detach().cpu().reshape(1, -1)
        
        # Save as a numpy array for storage in the DataFrame, including pred_recon
        data_rows.append({
            "Label": Label,
            "Index": w_im_idx,
            "pred_recon": pred_recon,
            "CF": w_cf_sklearn.numpy()
        })
        df = pd.DataFrame(data_rows)
        df.to_pickle(os.path.join(outputDir, "cf_vectors.pkl")

        w_im_idx += 1
        n_cfs += 1