import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
import numpy as np

import importlib
import warnings
import pickle
import os
import pandas as pd
import sys

from SimpleCNNarchitecture import SimpleCNN, evaluate
from helper import ImageDatasetWithLabels

# CelebA images and neural network predictors
# celebaDir = /PATH/TO/CELEBA/DATA
celebaDir = os.path.expanduser("~/Counterfactuals/celeba")
imDir = os.path.join(celebaDir, 'img_align_celeba') # Images themselves
predictorDir = os.path.join(celebaDir, 'predictors')
attr = pd.read_csv(os.path.join(celebaDir, 'list_attr_celeba.txt'), sep='\s+', header=1)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop((178, 178)),
    torchvision.transforms.Resize((256, 256))
])

if len(sys.argv) != 2:
    print("Usage: python label_arg.py <LabelName>")
    sys.exit(1)

Label = sys.argv[1]
print(f"Label: {Label}")

dataset = ImageDatasetWithLabels(imDir, attr, Label, transform=transform)

# Split dataset 80-20 into train and test
torch.manual_seed(1)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Initialize model, loss, and optimizer
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model = SimpleCNN().to(device)
criterion = nn.BCEWithLogitsLoss()  # binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print_interval = 250 # 100
model.train()

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    running_loss = 0.0
    total_correct = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        preds = torch.sigmoid(outputs) >= 0.5
        total_correct += torch.sum(preds == labels)

        # Print update every `print_interval` batches
        if (batch_idx + 1) % print_interval == 0:
            current_loss = loss.item()
            current_acc = torch.sum(preds==labels).item()/len(preds)
            print(f'Batch {batch_idx + 1}/{len(train_loader)} - Loss: {current_loss:.4f}, Accuracy: {current_acc:.4f}')

# Save model
fname = Label.lower() + '_weights.pt'
weights_path = os.path.join(predictorDir, fname)
torch.save(model.state_dict(), weights_path)

# Evaluate model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(weights_path))
model.eval()

subset_size = 1000
test_data_subset = Subset(test_loader.dataset, list(range(subset_size)))
test_loader_subset = DataLoader(test_data_subset, batch_size=test_loader.batch_size, shuffle=False)
_ = evaluate(model, test_loader_subset, criterion, device)