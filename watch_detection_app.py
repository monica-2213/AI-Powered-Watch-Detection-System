import os
import gdown
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Function to download a model from Google Drive
def download_model_from_gdrive(file_id, destination_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, destination_path, quiet=False)

# Google Drive file ID for your model
model_url = 'https://drive.google.com/uc?id=1-3SRZig9bvYvUYQmIVjO54l2oLlafpG4'
file_id = model_url.split('=')[-1]
model_path = 'unet-non-aug.keras'

# Download the model if not already present
if not os.path.exists(model_path):
    download_model_from_gdrive(file_id, model_path)

# Load the model
model = load_model(model_path)

def preprocess_image(image):
    image = cv2.resize(image, (512, 512))  # Resize to match model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def upload_and_predict(image_file):
    image_data = image_file.read()
    np_arr = np.frombuffer(image_data, np.uint8)
    original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    input_image = preprocess_image(original_image)
    predicted_mask = model.predict(input_image)

    # Process the output mask (assuming binary mask)
    predicted_mask = predicted_mask[0].squeeze()
    threshold = 0.1
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_background = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    color_segmented_image = np.where(binary_mask_resized[..., None] == 1, original_image, grayscale_background)

    # Display images
    st.image(original_image, caption='Original Image', use_column_width=True)
    st.image(binary_mask_resized, caption='Predicted Mask', use_column_width=True)
    st.image(color_segmented_image, caption='Segmented Image', use_column_width=True)

# Streamlit UI for file upload
st.title("AI Powered Watch Detection System")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    upload_and_predict(uploaded_file)
