import streamlit as st
import tensorflow as tf
import gdown
import zipfile
import os
import numpy as np
from PIL import Image

# Function to download and unzip the model file
def download_and_extract_model():
    # Google Drive file ID
    model_id = '1EAqFoUexroNtxyrxNXB_eoWYPTRPJw8c'
    # URL to download the model zip file
    url = f'https://drive.google.com/uc?id={model_id}'
    
    # Path to save the zip file
    zip_path = 'unet-non-aug.zip'
    # Path to extract the model file
    model_path = 'unet-non-aug.keras'

    # Download the zip file if it does not already exist
    if not os.path.exists(zip_path):
        st.write("Downloading model zip file...")
        gdown.download(url, zip_path, quiet=False)
    
    # Extract the zip file if the .keras model file does not exist
    if not os.path.exists(model_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()
            st.write(f"Model extracted to: {model_path}")
    
    # Check if the model file exists after extraction
    if not os.path.exists(model_path):
        st.error(f"Model extraction failed! Could not find '{model_path}'")
        return None
    
    st.write(f"Model file found at: {model_path}")
    return model_path

# Load the pre-trained U-Net model
def load_model():
    model_path = download_and_extract_model()
    if model_path is None:
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

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
        if model is None:
            st.error("Failed to load the model. Please check the model file.")
            return

        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Segment the image using the U-Net model
        segmented_image = segment_image(image, model)

        # Display the segmented result
        st.write("Segmented Image:")
        show_image(segmented_image)

if __name__ == "__main__":
    main()
