################ CELEBA DATASET ################
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageDatasetWithLabels(Dataset):
    def __init__(self, directory, label_frame, label_column, transform=None):
        """
        Initializes the dataset.
        :param directory: Path to the directory containing images.
        :param label_frame: A pandas DataFrame containing the labels.
        :param label_column: The column in label_frame that contains the labels.
        :param transform: Transformations to be applied to the images.
        """
        self.image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')])
        # Assumes index alignment with image files
        self.labels = label_frame[label_column].values==1
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Load label
        label = self.labels[idx]
        
        return image, label





################ COUNTERFACTUAL FUNCTIONS ################
import torch
import torch.nn as nn
import numpy as np


def latent_to_image(generator, latent):
    '''
    Passes latent vector in Style Space (W) through generator. 
    Produces an unclamped image in roughly 0-1 range.
    '''
    # Generate synthetic image
    if latent.ndim==2:
        latent = latent.unsqueeze(0)
    image_recon = generator.synthesis(latent)#, noise_mode='const'

    # Post-process so in 0-1 range (will need to clamp)
    image_recon = image_recon.add(1).div(2)
    return image_recon


def generate_counterfactual(generator, latent_vector, classifier, threshold, 
                            binary=False, Class=None, direction='increasing',
                            learning_rate=0.01, max_n_iter=500, verbose=False):
    assert direction in ['increasing', 'decreasing'], "Direction must be 'increasing' or 'decreasing'"
    sign = -1 if direction == 'increasing' else 1

    latent_vec = latent_vector.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([latent_vec], lr=learning_rate)
    probs = []
    
    for i in range(max_n_iter):
        # Generate synthetic image
        latent_stack = torch.stack([latent_vec] * generator.num_ws)
        image_recon = latent_to_image(generator, latent_stack)
        
        # Get prediction on class of interest
        if binary:
            predicted_prob = torch.sigmoid(classifier(image_recon))
        else:
            softmax = nn.Softmax(dim=1)
            predictions = softmax(classifier(image_recon))
            predicted_prob = predictions[0, Class]

        probs.append(round(predicted_prob.item(), 2))
        
        # Check convergence
        if (direction == 'increasing' and predicted_prob > threshold) or \
           (direction == 'decreasing' and predicted_prob < threshold):
            if verbose:
                print(i, probs)
            # # Clean up before break
            # del latent_stack, image_recon, predicted_prob
            # torch.cuda.empty_cache()
            break

        # Backpropagate
        optimizer.zero_grad()
        (sign * predicted_prob).backward()
        if latent_vec.grad.sum() == 0:
            print("Zero gradient encountered.")
            return None
        optimizer.step()
        
        if verbose and (i + 1) % 10 == 0:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            # print(f"Memory allocated: {torch.cuda.memory_allocated(device)/(2**30):.2f} GB")
            # print(f"Memory reserved: {torch.cuda.memory_reserved(device)/(2**30):.2f} GB")
            print(f"Iteration {i+1}/{max_n_iter}, Prediction: {predicted_prob.item():.3f}")
        
        # # Free intermediate variables and clear cache
        # del latent_stack, image_recon, predicted_prob
        # torch.cuda.empty_cache()
        
    if i == max_n_iter - 1:
        print("Failed to converge.")
    return latent_vec