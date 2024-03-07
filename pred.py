import os
import re
import glob
import matplotlib
import tensorflow as tf
from keras.models import load_model
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

# Set the path to the input image folder
input_folder = '/home/alireza/datasets/CARLA/test/raw'
output_folder = '/home/alireza/datasets/CARLA/test/depth/model03'

# Load the model
model_path = 'fine_tuned_model03.h5'  # Adjust the path to your model file
# model = load_model(model_path, compile=False)

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': tf.keras.layers.UpSampling2D, 'depth_loss_function': None}
model = load_model(model_path, custom_objects=custom_objects, compile=False)

# Iterate over all image files in the input folder
for input_file in glob.glob(os.path.join(input_folder, '*.png')):
    # Load input image
    input_image = load_images([input_file])
    
    # Predict depth
    outputs = predict(model, input_image)
    os.makedirs(output_folder, exist_ok=True)
    
    # Save output image with the same name as the input image
    # Extract numeric part from the input filename
    numeric_part = re.findall(r'\d+', input_file)[0]
    output_file = os.path.join(output_folder, f"depth_img_{numeric_part}.png")
    
    # output_file = os.path.splitext(input_file)[0] + '-depth.png'
    viz = display_images(outputs.copy())
    plt.imsave(output_file,viz)
    plt.figure(figsize=(10,5))
    plt.imshow(viz)
    # plt.savefig(output_file)
    plt.show()
