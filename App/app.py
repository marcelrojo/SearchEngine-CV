import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import cv2
from PIL import Image as PILImage
import glob
import os
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURABLE PARAMETERS ---
IMAGE_SIZE = (256, 256)
EMBEDDING_DIMENSIONS = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    {"name": "ResNET50", "model_path": os.path.join(BASE_DIR, "src", "Models", "ResNet50.keras"), "csv_path": os.path.join(BASE_DIR, "src", "CSVs", "ResNet50_pure_new.csv")},
    {"name": "Classification Model", "model_path": os.path.join(BASE_DIR, "src", "Models", "classification_model.keras"), "csv_path": os.path.join(BASE_DIR, "src", "CSVs", "classification_model_embeddings.csv")},
    {"name": "ResNET50 retrained", "model_path": os.path.join(BASE_DIR, "src", "Models", "ResNet50_retrained.keras"), "csv_path": os.path.join(BASE_DIR, "src", "CSVs", "ResNet50_retrained.csv")},
]

DATA_FOLDER = os.path.join(BASE_DIR, "src", "Data", "*")

# --- CUSTOM LAYER FOR MODEL ---
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    
# --- HELPER FUNCTIONS ---
def read_image_og(file_path):
    """Read and preprocess the image."""
    img = PILImage.open(file_path)
    img = np.array(img)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imshow(image_array):
    """Convert an image array to a PIL image for Streamlit display."""
    image_array = image_array.clip(0, 255).astype("uint8")
    return PILImage.fromarray(image_array)

# ---Read image for prediction
def resize_crop(image, image_size=(256, 256)):
    target_height, target_width = image_size
    original_height, original_width = image.shape[:2]

    original_aspect = original_width / original_height
    target_aspect = target_width / target_height

    if original_aspect > target_aspect:
        new_width = original_height
        crop_x = (original_width - new_width) // 2
        cropped_image = image[:, crop_x:crop_x + new_width]
    elif original_aspect < target_aspect:
        new_height = original_width
        crop_y = (original_height - new_height) // 2
        cropped_image = image[crop_y:crop_y + new_height, :]
    else:
        cropped_image = image

    resized_image = cv2.resize(cropped_image, (target_width, target_height))
    return resized_image

def preprocess(image):
    image = image / 255.0
    image = resize_crop(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def read_image_preprocessed(image_file):
    image = PILImage.open(image_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = preprocess(image) 
    return image    

# --- LOAD MODEL AND EMBEDDINGS ---
@st.cache_resource
def load_model_and_csv(model_info):
    """Load the model and Annoy index."""
    model = load_model(model_info["model_path"], custom_objects={"L2Normalization": L2Normalization})
    embeddings_df = pd.read_csv(model_info["csv_path"])
    return model, embeddings_df

# --- GET MOST SIMILAR MATCHES ---
def get_most_similar_images(query_image_path, embedding_model, embeddings_df, k, preprocessed=True):
    if preprocessed: query_image = read_image_preprocessed(query_image_path)
    else: query_image = read_image_og(query_image_path)
    
    query_image = np.expand_dims(query_image, axis=0)
    
    query_embedding = embedding_model.predict(query_image).flatten()
    
    embeddings = embeddings_df.drop(columns=['filename']).values
    filenames = embeddings_df['filename'].values
    
    similarities = cosine_similarity([query_embedding], embeddings)
    
    top_k_indices = similarities[0].argsort()[-k:][::-1]
    
    most_similar_filenames = filenames[top_k_indices]
    
    return most_similar_filenames, similarities[0][top_k_indices]

# --- STREAMLIT APP ---
st.title("Image Search Engine")
st.write("Upload an image to find the most similar images.")

model_choice = st.selectbox("Select a model", [model["name"] for model in MODELS])
selected_model_info = next(model for model in MODELS if model["name"] == model_choice)
model, csv = load_model_and_csv(selected_model_info)


# File uploader for query images
uploaded_file = st.file_uploader("Choose a query image", type=["jpg", "png"])

if uploaded_file:

    # --- READ AND PROCESS QUERY IMAGE ---
    query_image = PILImage.open(uploaded_file).convert("RGB")
    query_array = np.array(query_image)
    query_resized = cv2.resize(query_array, IMAGE_SIZE)
    
    preprocessed = True
    
    if selected_model_info["name"] == "ResNET50":
        preprocessed = False
    
    # --- GET SIMILAR IMAGES ---
    similar_files, distances = get_most_similar_images(uploaded_file, model, csv, 3, preprocessed=preprocessed)
    
    # --- DISPLAY QUERY IMAGE ---
    st.subheader("Query Image")
    st.image(imshow(query_resized), caption="Query Image")
    
    # --- DISPLAY RESULTS ---
    st.subheader("Top Matches")
    image_files = glob.glob(os.path.join(DATA_FOLDER, "*.jpg"), recursive=True)
    
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i < len(similar_files):
            match_image = read_image_og(f"{BASE_DIR}/src/Data/" + similar_files[i])
            match_image = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
            col.image(imshow(match_image), caption=f"Distance: {distances[i]:.4f}", use_container_width=True)