
import os
import cv2
import torch
from torchsr.models import ninasr_b2
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

input_folder = '/home/alireza/datasets/CARLA/test/depth/model03'
output_folder = '/home/alireza/datasets/CARLA/test/depth/model03/sr_ninab2/gpu'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize the model on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ninasr_b2(scale=2, pretrained=True).to(device)
counter = 0
# List all image filenames
filenames = [filename for filename in os.listdir(input_folder) if filename.endswith(('.png', '.jpg', '.jpeg'))]

for filename in filenames:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    # Load the image
    lr = cv2.imread(input_path)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and move to GPU
    lr_t = to_tensor(lr).unsqueeze(0).to(device)
    
    # Apply super-resolution
    with torch.no_grad():
        sr_t = model(lr_t)
    
    # Move tensor back to CPU and save the super-resolved image
    save_image(sr_t.cpu().squeeze(), output_path)
    counter = counter + 1
    print(counter)

print("Super-resolution completed.")
