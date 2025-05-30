import torch
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle



# CelebA images and neural network predictors
# celebaDir = /PATH/TO/CELEBA/DATA
cfDir = os.path.expanduser("~/Counterfactuals")
celebaDir = os.path.join(cfDir, 'celeba')
# styleganDir = /PATH/TO/STYLEGAN/REPO
styleganDir = os.path.join(cfDir, 'stylegan3-projector')

import subprocess
git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
outputDir = os.path.join(git_root, "CelebA", "Data")

attr = pd.read_csv(os.path.join(celebaDir,'list_attr_celeba.txt'), sep='\s+', header=1)


# Files are first 10k .jpgs, in order
w_fname_all = os.path.join(outputDir, 'w_all_10k.pt')
X = torch.load(w_fname_all).detach().numpy()

# Modify attribute labels to simplify importance scores
facial_hair = ((attr.Sideburns == 1) | (attr.Goatee == 1) | 
               (attr.No_Beard == -1) | (attr['5_o_Clock_Shadow']==1) | 
              (attr.Mustache == 1)).astype(int)
dark_hair = ((attr.Black_Hair == 1) | (attr.Brown_Hair==1) |
             (attr.Blond_Hair==-1)).astype(int)

overweight = ((attr.Chubby==1) | (attr.Double_Chin==1)).astype(int)

# abstract_cols = ['Attractive', 'Blurry', 'Young', 'Male', 'Mouth_Slightly_Open']
abstract_cols = ['Blurry', 'Mouth_Slightly_Open']
redundant_cols = ['Gray_Hair', 'Blond_Hair', 'Black_Hair', 'Brown_Hair',
                  'Sideburns', 'Goatee', 'No_Beard', '5_o_Clock_Shadow', 'Mustache',
                 'Chubby', 'Double_Chin']
exclude_cols = abstract_cols + redundant_cols

clean_Ys = pd.concat([attr.loc[:, ~attr.columns.isin(exclude_cols)],
                    facial_hair.rename("Facial_hair"),
                    dark_hair.rename("Dark_Hair"),
                    overweight.rename("Overweight")], axis=1).iloc[:X.shape[0]]

clean_Ys[clean_Ys==-1] = 0
clean_Ys.head()

logregs_path = os.path.join(outputDir, 'attribute_logregs.pkl')

logregs = {}
# for column in all_Ys.columns:
for column in clean_Ys.columns:
    # Create logistic regression model
    logreg = LogisticRegression()
    
    # logreg.fit(X, all_Ys[column])
    logreg.fit(X, clean_Ys[column])
    logregs[column] = logreg

with open(logregs_path, 'wb') as file:
    pickle.dump(logregs, file)
