import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import cv2
from PIL import Image as PILImage
import glob
import os
from annoy import AnnoyIndex
import matplotlib.pyplot as plt

# --- CONFIGURABLE PARAMETERS ---
IMAGE_SIZE = (256, 256)
EMBEDDING_DIMENSIONS = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "Models", "transfer_custom_big.keras")
ANNOY_PATH = os.path.join(BASE_DIR, "src", "Annoys", "custom_transfer_big.ann")
DATA_FOLDER = os.path.join(BASE_DIR, "src", "Data", "*")

# --- CUSTOM LAYER FOR MODEL ---
class L2Normalization(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    
# --- HELPER FUNCTIONS ---
def read_image(file_path):
    """Read and preprocess the image."""
    img = cv2.imread(file_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imshow(image_array):
    """Convert an image array to a PIL image for Streamlit display."""
    image_array = image_array.clip(0, 255).astype("uint8")
    return PILImage.fromarray(image_array)

# --- LOAD MODEL AND ANN ---
@st.cache_resource
def load_model_and_ann():
    """Load the model and Annoy index."""
    model = load_model(MODEL_PATH, custom_objects={"L2Normalization": L2Normalization})
    index = AnnoyIndex(EMBEDDING_DIMENSIONS, "angular")
    index.load(ANNOY_PATH)
    return model, index

# --- STREAMLIT APP ---
st.title("Image Search Engine")
st.write("Upload an image to find the most similar images.")

# File uploader for query images
uploaded_file = st.file_uploader("Choose a query image", type=["jpg", "png"])

if uploaded_file:
    # --- LOAD MODEL & ANN ---
    model, index = load_model_and_ann()
    
    # --- READ AND PROCESS QUERY IMAGE ---
    query_image = PILImage.open(uploaded_file).convert("RGB")
    query_array = np.array(query_image)
    query_resized = cv2.resize(query_array, IMAGE_SIZE)
    
    # --- GET EMBEDDINGS ---
    embedding = model.predict(np.expand_dims(query_resized, axis=0))[0]
    
    # --- FIND NEAREST NEIGHBORS ---
    similar_indices, distances = index.get_nns_by_vector(embedding, 3, include_distances=True)
    
    # --- DISPLAY QUERY IMAGE ---
    st.subheader("Query Image")
    st.image(imshow(query_resized), caption="Query Image")
    
    # --- DISPLAY RESULTS ---
    st.subheader("Top Matches")
    image_files = glob.glob(os.path.join(DATA_FOLDER, "*.jpg"), recursive=True)
    
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if i < len(similar_indices):
            #match_path = image_files[similar_indices[i]]
            match_path = image_files[0]
            match_image = read_image(match_path)
            col.image(imshow(match_image), caption=f"Distance: {distances[i]:.4f}", use_container_width=True)