import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import os

import sys
cfDir = "/accounts/grad/jeremy_goldwasser/Counterfactuals/"
styleganDir = cfDir + 'stylegan3-projector/'
sys.path.append(styleganDir)

import torch_utils
import dnnlib
import pickle

from SimpleCNNarchitecture import *
import projection
import helper
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataDir = "/accounts/grad/jeremy_goldwasser/Counterfactuals/celeba/"
imDir = dataDir + 'img_align_celeba/'
projectionDir = os.path.join(cfDir, 'cf_stylegan_celeba', 'projections')

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd

to_pil = transforms.ToPILImage()
os.makedirs("cf_vecs", exist_ok=True)

# Evaluate model performance on subset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((178, 178)),
    transforms.Resize((256, 256))
])

celebaDir = cfDir + "celeba/"
predDir = celebaDir + "predictors/"

path = styleganDir+'results/network-snapshot-008800.pkl'
with open(path, 'rb') as fp:
    G = pickle.load(fp).to(device)

Labels = ['Young', 'Attractive', 'Male', 'Smiling']
# try_idx = [3, 5, 21, 27, 30, 32, 35, 39, 40, 46, 47, 50, 56]
n_for_cf = 100
upper_thresh = 0.75; lower_thresh = 0.25
# upper_thresh = 0.5; lower_thresh = 0.5

data_rows = []
for Label in Labels:
    weights_path = predDir + Label.lower() + '_weights.pt'
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    n_cfs = 0
    w_im_idx = 0
    # for w_im_idx in range(n_for_cf):
    while n_cfs < n_for_cf:
        print(f"{Label} index {w_im_idx}")
        w_fnames = os.listdir(projectionDir)
        w_fname = w_fnames[w_im_idx]
        
        # Load image
        idx_fname = w_fname.split('.')[0].split('_')[1]
        num_fname = str(int(idx_fname)+1)
        num_zeros_needed = 6 - len(num_fname)
        padded_num_fname = '0' * num_zeros_needed + num_fname
        jpg_fname = padded_num_fname + '.jpg'
        image = Image.open(imDir + jpg_fname)
        
        # Reconstructed image
        w_vec = torch.load(os.path.join(projectionDir, w_fname)).to(device)
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
        df.to_pickle("cf_vecs/cf_vectors.pkl")

        w_im_idx += 1
        n_cfs += 1