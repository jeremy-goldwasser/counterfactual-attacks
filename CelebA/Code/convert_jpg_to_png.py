import os
from PIL import Image
import torch
from torchvision import transforms

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((178, 178)),
    transforms.Resize((256, 256))
])

# Paths
input_dir = "/accounts/grad/jeremy_goldwasser/Counterfactuals/celeba/img_align_celeba"
output_dir = "/var/tmp/celeba/img_align_celeba_png"

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Process images
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        img = transforms.ToPILImage()(img)

        new_filename = os.path.splitext(filename)[0] + ".png"
        img.save(os.path.join(output_dir, new_filename))
