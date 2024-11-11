import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# Function to download the pre-trained model from Google Drive
def download_model():
    model_url = "https://drive.google.com/file/d/1-3SRZig9bvYvUYQmIVjO54l2oLlafpG4/view?usp=sharing"  # Replace with your Google Drive model link
    model_path = "unet-non-aug.keras"
    
    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)
    
    return model_path

# Load your pre-trained U-Net model
model_path = download_model()
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('RGB')
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (256, 256))  # Assuming your model accepts 256x256
    img_resized = img_resized / 255.0  # Normalize the image
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension
    return img_resized

# Function to segment the image
def segment_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)  # Get prediction from model
    prediction = (prediction > 0.5).astype(np.uint8)  # Thresholding
    
    mask = prediction[0, :, :, 0]  # Assuming single-channel output (for binary mask)
    original_size = image.size
    mask_resized = cv2.resize(mask, (original_size[0], original_size[1]), interpolation=cv2.INTER_NEAREST)

    return mask_resized

# Function to blend the results (colorized watch area with grayscale background)
def blend_image(image, mask):
    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    
    color_watch = np.zeros_like(img_array)
    color_watch[:, :, :] = img_array[:, :, :]
    
    color_watch[mask == 1] = img_array[mask == 1]
    gray_image[mask == 1] = color_watch[mask == 1]
    
    return gray_image

# Streamlit UI setup
st.set_page_config(page_title="AI-Powered Watch Detection", layout="centered")

# Custom CSS for better UI design
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .upload-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }
        .uploaded-image {
            border-radius: 10px;
            margin-top: 20px;
        }
        .output-title {
            font-size: 1.5em;
            font-weight: bold;
            color: #4CAF50;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the web app
st.markdown('<div class="title">AI-Powered Watch Detection System</div>', unsafe_allow_html=True)

# Upload image section
with st.container():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True, class_="uploaded-image")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_image is not None:
    # Segment the image
    mask = segment_image(image)

    # Blend the segmented output
    blended_image = blend_image(image, mask)

    # Convert blended image to PIL for display
    blended_image_pil = Image.fromarray(blended_image)

    # Output the segmented result
    st.markdown('<div class="output-title">Segmented Output</div>', unsafe_allow_html=True)
    st.image(blended_image_pil, caption='Segmented Watch', use_column_width=True)
