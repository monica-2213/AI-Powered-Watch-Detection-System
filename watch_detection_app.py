import streamlit as st
import tensorflow as tf
import gdown
import os
import numpy as np
from PIL import Image
import zipfile

# Download and unzip the model from Google Drive
def download_model():
    # Google Drive file ID (updated based on the new link)
    model_id = '1Q97oFaLn8aC8aKJjkZQ-WIlJiLjKUQO1'
    # URL to download the model
    url = f'https://drive.google.com/uc?id={model_id}'
    
    # Path to save the model zip file
    model_zip_path = 'unet_model.zip'

    # Download the model zip if not already present
    if not os.path.exists(model_zip_path):
        st.write(f"Downloading model from: {url}")
        gdown.download(url, model_zip_path, quiet=False)
    
    # Extract the zip file if it's not already extracted
    model_path = 'unet-non-aug.keras'
    if not os.path.exists(model_path):
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall()
    
    # Verify if the model path exists and is valid
    if not os.path.exists(model_path):
        st.write("Model file does not exist or extraction failed.")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return model_path

# Load the pre-trained U-Net model
def load_model():
    model_path = download_model()
    st.write(f"Attempting to load model from: {model_path}")
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            st.write("Model loaded successfully.")
            return model
        except Exception as e:
            st.write(f"Error loading the model: {str(e)}")
            raise ValueError(f"Error loading the model: {str(e)}")
    else:
        st.write(f"Model path {model_path} does not exist.")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
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
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

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
