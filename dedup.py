from towhee import pipeline
import glob
import os
import numpy as np
try:
  # os.chdir("path/of/the/directory")
  #Fetching the towhee image-encoding pipline
  embedding_pipeline = pipeline('image-encoding')
  #getting all the images
  png = glob.glob('*.png')
  jpg = glob.glob('*.jpg')
  jpeg = glob.glob('*.jpeg')
  all_images=jpeg+jpg+png
  total=len(all_images)
  dub_imaged=[]
  #printing all the images in the folder
  print("All images in the folder ",all_images)
  for i in range (0,total-1):
    image= embedding_pipeline(all_images[i]) # getting embeddings
    for j in range(i+1,total-1):
      img1_embedding= embedding_pipeline(all_images[j]) 
      is_sim = np.linalg.norm(image - img1_embedding) < 0.01 #comparing the embeddings
      if (is_sim):
        dub_imaged.append(all_images[j])#adding the dublicate images into the dublicate array
  #print all the dublicate images
  print("Dublicate Images",dub_imaged)
  for photo in dub_imaged:
    os.remove(photo)#removing dublicate images 
  print("Removed All Dublicates")  
except Exception as e:
  print(e)