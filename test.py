import torch
import requests 
print(torch.__version__)
print(torch.cuda.is_available())
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
# from PIL import Image
# image_url= "https://upload.wikimedia.org/wikipedia/commons/9/9a/Cape_may.jpg"
# # download and show image
# # img = PIL.Image.open(image_url)
# data=requests.get(image_url).content

# f = open('img.jpg','wb')
# f.write(data)
# f.close()
# img = Image.open(r"img.jpg")
# img.show()     
# 

import clip

import os
from PIL import Image
import numpy as np
model = SentenceTransformer('clip-ViT-B-32')

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)

# Define the directory path containing the images
image_directory = 'flickr30k_images/flickr30k_images'

# Initialize an empty list to store image embeddings and metadata
image_data = []

# Loop through the files in the directory
for filename in tqdm(os.listdir(image_directory), desc="Processing images", unit="image"):
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_directory, filename)
        img = Image.open(img_path)
        
        # Encode the image using the model
        img_emb = model.encode(img)
        # img_emb /= img_emb.norm(dim=-1, keepdim=True)
        
        # Create a tuple with identifier, vector, and metadata
        image_tuple = (
            filename,
            img_emb,
            {"type": "jpg"}
        )
        
        # Append the tuple to the image_data list
        image_data.append(image_tuple)