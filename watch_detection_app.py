import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import io
import matplotlib.pyplot as plt

# Load the pre-trained U-Net model from your Google Drive (replace with correct path)
MODEL_PATH = '/content/drive/MyDrive/27_Oct_2024_Dataset_Creation/StratifiedDataset/newfiles_11Nov/unet-non-aug.keras'  # Example: '/content/drive/MyDrive/model/unet_watch_detection.h5'
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    # Resize the image to 512x512
    image = image.resize((512, 512))
    # Convert image to numpy array
    image_array = np.array(image)
    # Normalize the image to [0, 1] for the model
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def segment_image(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make prediction with the model
    prediction = model.predict(preprocessed_image)
    
    # Squeeze the batch dimension
    prediction = np.squeeze(prediction, axis=0)
    
    # Create the output image: color watch, grayscale background
    # Convert the prediction to a binary mask (thresholding)
    mask = prediction > 0.5  # Assuming binary segmentation
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension for blending
    
    # Create the original image and grayscale background
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to 3 channels
    
    # Apply the mask to retain the color on the watch and grayscale on the background
    result = np.where(mask == 1, image_array, gray_image)
    
    return result

def show_image(image):
    # Display image using matplotlib (streamlit might not render images correctly otherwise)
    st.image(image, channels="RGB", use_column_width=True)

def main():
    st.title("Watch Detection App")
    
    st.write("Upload an image to detect and segment the watch:")
    
    # Upload file widget
    uploaded_file = st.file_uploader("Choose an image...", type="jpg, jpeg, png")
    
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Segment the image using the U-Net model
        segmented_image = segment_image(image)
        
        # Display the segmented result
        st.write("Segmented Image:")
        show_image(segmented_image)

if __name__ == "__main__":
    main()
