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

model_ID = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_ID)
preprocess = CLIPImageProcessor.from_pretrained(model_ID)


def load_and_preprocess_image_cv(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(image, return_tensors="pt")
    return image

def random_images_to_display(image_files, num_images=10):
    return random.sample(image_files, num_images)

def emb_count(image_files):
    image_embeddings = []

    for image_file in image_files:
        try:
            image_path = os.path.join(image_folder, image_file)
            image = load_and_preprocess_image_cv(image_path)["pixel_values"]
            with torch.no_grad():
                image_embedding = model.get_image_features(image)
            image_embeddings.append(image_embedding)
        except:
            continue
    return image_embeddings

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

def sample_images(row, image_files, n=250):
    return random.sample(image_files, n)