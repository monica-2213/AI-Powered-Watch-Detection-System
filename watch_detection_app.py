import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Download the model from Google Drive
def download_model():
    # Google Drive file ID
    model_id = '1-3SRZig9bvYvUYQmIVjO54l2oLlafpG4'
    # URL to download the model
    url = f'https://drive.google.com/uc?id={model_id}'
    
    # Path to save the model
    model_path = 'unet-non-aug.keras'

    # Download the model if it is not already present
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    return model_path

# Load the pre-trained U-Net model
def load_model():
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image):
    """
    Preprocess the uploaded image:
    - Resize to 512x512
    - Normalize to [0, 1] range
    - Add batch dimension
    """
    image = image.resize((512, 512))  # Resize the image to 512x512
    image_array = np.array(image)  # Convert the image to a numpy array
    image_array = image_array / 255.0  # Normalize the image to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def segment_image(image, model):
    """
    Segment the uploaded image using the pre-trained U-Net model.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction with the model
    prediction = model.predict(preprocessed_image)
    
    # Squeeze the batch dimension
    prediction = np.squeeze(prediction, axis=0)
    
    # Create the output image: color watch, grayscale background
    mask = prediction > 0.5  # Assuming binary segmentation
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension for blending
    
    # Create grayscale image without using OpenCV
    gray_image = np.array(image.convert('L'))  # Convert to grayscale
    gray_image = np.stack([gray_image] * 3, axis=-1)  # Make it 3 channels for compatibility
    
    # Apply the mask to retain the color on the watch and grayscale on the background
    result = np.where(mask == 1, np.array(image), gray_image)
    
    return result

def show_image(image):
    """
    Display the image in Streamlit using the proper color channels.
    """
    st.image(image, channels="RGB", use_column_width=True)

def main():
    st.title("Watch Detection App")

    st.write("Upload an image to detect and segment the watch:")

    # Upload file widget
    uploaded_file = st.file_uploader("Choose an image...", type="jpg, jpeg, png")

    if uploaded_file is not None:
        # Load the model
        model = load_model()

        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Segment the image using the U-Net model
        segmented_image = segment_image(image, model)

        # Display the segmented result
        st.write("Segmented Image:")
        show_image(segmented_image)

if __name__ == "__main__":
    main()
