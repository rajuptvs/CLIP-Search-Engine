from torch.utils.data import Dataset
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import clip
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import vecs


class ClipSearchDataset(Dataset):
    def __init__(self, img_dir,  img_ext_list = ['.jpg', '.png', '.jpeg', '.tiff'], preprocess = None):    
        self.preprocess = preprocess
        self.img_path_list = []
        self.walk_dir(img_dir, img_ext_list)
        print(f'Found {len(self.img_path_list)} images in {img_dir}')

    def walk_dir(self, dir_path, img_ext_list): # work for symbolic link
        for root, dirs, files in os.walk(dir_path):
            self.img_path_list.extend(
                os.path.join(root, file) for file in files 
                if os.path.splitext(file)[1].lower() in img_ext_list
            )
            
            for dir in dirs:
                full_dir_path = os.path.join(root, dir)
                if os.path.islink(full_dir_path):
                    self.walk_dir(full_dir_path, img_ext_list)

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        return img, img_path


def compute_embeddings(img_dir, save_path, batch_size, num_workers, vector_db=False,DB_CONNECTION=None):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    dataset = ClipSearchDataset(img_dir = img_dir, preprocess = preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    img_path_list, embedding_list = [], []
    for img, img_path in tqdm(dataloader):
        with torch.no_grad():
            features = model.encode_image(img.to(device))
            features /= features.norm(dim=-1, keepdim=True)
            embedding_list.extend(features.detach().cpu().numpy())
            img_path_list.extend(img_path)

    result = {'img_path': img_path_list, 'embedding': embedding_list}


    if vector_db:
        print("Creating / connecting vector_db")
        vx = vecs.create_client(DB_CONNECTION)
        images = vx.get_or_create_collection(name="vector_db", dimension=512)
        image_data = [
            (
                result['img_path'][i],
                result['embedding'][i],
                {"type": "jpg"}
            )
            for i in range(len(result['embedding']))
        ]
        print("Uploading vectors")
        images.upsert(image_data)
        images.create_index()
        print("Done")

    else:
        with open(save_path, 'wb') as f:
            pickle.dump(result, f, protocol=4)


    return result

def create_faiss_index(embeddings_path, save_path):
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)

    embeddings = np.array(results['embedding'], dtype=np.float32)

    index = faiss.index_factory(embeddings.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(embeddings)
    # save index
    faiss.write_index(index, save_path)

if __name__ == "__main__":
    vector_db = True
    test=compute_embeddings(img_dir="flickr30k_images/flickr30k_images", save_path="test/embeddings.pkl",batch_size=32,num_workers=20,vector_db=vector_db,DB_CONNECTION="postgresql://postgres:Heythereitsme123@db.hwasuqmoxribpmmwjmra.supabase.co:5432/postgres")
    #only if not using vector_db
    if not vector_db:
        create_faiss_index(embeddings_path="test/embeddings.pkl",save_path="index.faiss")

    # print(test)