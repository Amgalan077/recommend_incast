import time
import random
import os

from PIL import Image
import torch
import cv2
import faiss
import numpy as np
import pandas as pd
from transformers import CLIPImageProcessor, CLIPModel
import streamlit as st
import joblib

from utils.functions import load_and_preprocess_image_cv, random_images_to_display, emb_count, sample_images

st.markdown(
f'<style>div.stButton > button:first-child {{background-color: rgba(255, 255, 255, 0.5); border: 2px solid red;}}</style>',
unsafe_allow_html=True
)

model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)

image_folder = 'photos'
image_files = os.listdir(image_folder)

our_data = pd.read_csv('our_data.csv')
our_data['recommended'] = our_data.apply(lambda row: sample_images(row, image_files), axis=1)
df = pd.read_csv('df.csv')
df['Vector'] = df['Vector'].apply(lambda x: np.fromstring(x, sep=' '))
books_vector = np.loadtxt('vectors.txt')
index = faiss.IndexFlatIP(books_vector.shape[1])
index.add(books_vector)

image_embeddings = joblib.load('image_embeddings.joblib')

# def main():
#     liked_images = []
#     st.sidebar.title(liked_images)
#     initial_images = random_images_to_display(image_files, num_images=10)

#     for image_name in initial_images:
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(os.path.join(image_folder, image_name), use_column_width=True)
#         with col2:
#             button_key = f"like_button_{image_name}"
#             liked = st.button(f"Like", key=button_key)
#             if liked:
#                 liked_images.extend(liked)

#     if st.button("Recommend"):
#         recommended_images = recommendation(liked_images)
#         st.header("Recommended Images:")
#         for image_name in recommended_images:
#             st.image(os.path.join(image_folder, image_name), use_column_width=True, caption=image_name)

#     st.session_state.liked_images = liked_images

# if __name__ == "__main__":
#     main()

def recommendation(array: list, neighbours_clip: int = 17, neighbours_discription:int = 5) -> list:
    """
    array - list liked photo
    neighbours_clip - count predictions with CLIP
    neighbours_discription - count predictions with discription
    """
    recarray = []
    for img in array:
        path = os.path.join('photos', img)
        image = load_and_preprocess_image_cv(path)["pixel_values"]
        with torch.no_grad():
            embedding = model.get_image_features(image)
        similarity_scores = [torch.nn.functional.cosine_similarity(embedding, image_embedding) for image_embedding in image_embeddings]
        top_n_similarities, top_n_indices = torch.topk(torch.tensor(similarity_scores), k=neighbours_clip)
        top = np.array(image_files)[top_n_indices.tolist()].tolist()[1:]
        recarray.extend(top)
    for imgdis in array:
        user_text_pred = df[df['Image Name'] == imgdis]['Vector'].values[0]
        D, I = index.search(user_text_pred.reshape(1, -1), k=neighbours_discription)
        recarray.extend(df['Image Name'].iloc[I[0][1:]].values)
    
    recarray = list(set(recarray))
    return random.sample(recarray, 20)
@st.cache_data
def intersection(rec:list) -> int:
    rec = set(rec)
    listrec = []
    for i in our_data['recommended']:
        listrec.append(len(set(rec) & set(i)))
    most_relevant = np.array(listrec).argsort()[-1]
    return our_data.iloc[most_relevant]

if 'rec_images' not in st.session_state:
    st.session_state.rec_images = []
st.write(len(st.session_state.rec_images))
col1, col2 = st.columns(2)
with col1:
    if len(st.session_state.rec_images) == 0:
        image_name = random_images_to_display(image_files, num_images=1)
        st.image(os.path.join(image_folder, *image_name), use_column_width=True)
    else:
        image_name = random.sample(st.session_state.rec_images, 1)
        st.image(os.path.join(image_folder, *image_name), use_column_width=True)
with col2:
    liked = st.button("Like")
    skiped = st.button('Skip')
    if liked:
        st.session_state.rec_images.extend(recommendation(image_name))
    if skiped:
        pass
        

# st.write(rec_images)
# st.sidebar.title(len(rec_images))
st.sidebar.write(intersection(st.session_state.rec_images))
st.write(len(st.session_state.rec_images))
    