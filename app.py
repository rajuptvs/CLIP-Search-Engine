import streamlit as st
import os
from PIL import Image
import pickle
import faiss
import numpy as np
import clip
import torch
import requests 
import vecs
from sentence_transformers import SentenceTransformer
# import protocolbuffers
# protocolbuffers.Set_PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION("python")

st.set_page_config(layout="wide")

@st.cache_resource
def load_data(faiss_index_path, embeddings_path, device=0):
    # load faiss index
    index = faiss.read_index(faiss_index_path)
    # load embeddings
    with open(embeddings_path, 'rb') as f:
        results = pickle.load(f)
    embedding_path_list = results['img_path']
    model, preprocess = clip.load('ViT-B/32', device)
    return index, embedding_path_list, model, preprocess

# preprocess
device = 0

db = st.sidebar.selectbox('Storage', ('VectorDB', 'Local'))

if db == 'VectorDB':

    ###########################'
    # Needs attention for reformating
    ##############################
    vector_db_path = st.sidebar.text_input('VectorDB path',value="postgresql://postgres:Heythereitsme123@db.hwasuqmoxribpmmwjmra.supabase.co:5432/postgres")
    vx = vecs.create_client(vector_db_path)
    images_emb = vx.get_collection(name="vector_db")
    model, preprocess = clip.load('ViT-B/32', device)
    with open("test\embeddings.pkl", 'rb') as f:
        results = pickle.load(f)
    embedding_path_list = results['img_path']
else:
    faiss_index_path = 'index.faiss'
    embeddings_path = 'test/embeddings.pkl'
    index, embedding_path_list, model, preprocess = load_data(faiss_index_path, embeddings_path, device)

# select box
search_mode = st.sidebar.selectbox('Search mode', ('Image', 'Text'))

# sliders
if search_mode == 'Image':
    similar_type=st.sidebar.selectbox('Similarity type', ('URL', 'Image'))
    if similar_type == 'URL':
        image_url = st.sidebar.text_input('Image URL',value="https://upload.wikimedia.org/wikipedia/commons/9/9a/Cape_may.jpg")
        data=requests.get(image_url).content
        f = open('img.jpg','wb')
        f.write(data)
        f.close()
        img_path = 'img.jpg'
        
    else:
        img_idx = st.slider('Image index', 0, len(embedding_path_list)-1, 0)
        img_path = embedding_path_list[img_idx]
num_search = st.sidebar.slider('Number of search results', 1, 20, 5)
images_per_row = st.sidebar.slider('Images per row', 1, num_search, min(5, num_search))

if search_mode == 'Image':
    # search by image
    print(img_path)
    img = Image.open(img_path).convert('RGB')
    st.image(img, caption=f'Query Image: {img_path}')
    img_tensor = preprocess(img).unsqueeze(0).to(0)
    with torch.no_grad():
        features = model.encode_image(img_tensor.to(device))
else:
    # search by text
    query_text = st.text_input('Enter a search term:')
    with torch.no_grad():
        text = clip.tokenize([query_text]).to(device)
        features = model.encode_text(text)

features /= features.norm(dim=-1, keepdim=True)
embedding_query = features.detach().cpu().numpy().astype(np.float32)
if db == 'Local':
    D,I = index.search(embedding_query, num_search)
    print(D)
    print("@#@@@@")
    print(I)
    print("@@@@@@@@@@@@@")
else:
    model = SentenceTransformer('clip-ViT-B-32')
    embedding_query = model.encode(query_text)
    I = images_emb.query(
        embedding_query,
        limit=num_search,
        filters={"type": {"$eq": "jpg"}},  
    )
    print(I)
# print(embedding_path_list)
# print(type(embedding_path_list))
if db == 'Local':
    match_path_list = [embedding_path_list[i] for i in I[0]]
else:
    match_path_list = I
print(type(match_path_list))
print("matc")
print(match_path_list)
# calculate number of rows
num_rows = -(-num_search // images_per_row)  # Equivalent to ceil(num_search / images_per_row)

# display
for i in range(num_rows):
    cols = st.columns(images_per_row)
    for j in range(images_per_row):
        idx = i*images_per_row + j
        if idx < num_search:
            path = match_path_list[idx]
            
            img = Image.open(path).convert('RGB')
            if db == 'Local':
                cols[j].image(img, caption=f' path {path}', use_column_width=True)
            else:
                cols[j].image(img, caption=f'path {path}', use_column_width=True)