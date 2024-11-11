import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

# Load the pre-trained U-Net model from Google Drive (ensure the path is correct)
MODEL_PATH = 'unet-non-aug.keras'  # Update with your model path if necessary
model = tf.keras.models.load_model(MODEL_PATH)

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

def segment_image(image):
    """
    Segments the image using the pre-trained U-Net model:
    - Preprocess the image
    - Predict the mask (binary segmentation)
    - Apply mask to colorize the watch area and grayscale the background
    """
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)  # Make prediction with the model
    prediction = np.squeeze(prediction, axis=0)  # Remove batch dimension
    
    # Create binary mask from prediction (thresholding at 0.5)
    mask = prediction > 0.5
    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension for blending
    
    # Convert the input image to grayscale for background
    image_array = np.array(image)
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel grayscale
    
    # Apply the mask: retain color on the watch and grayscale the background
    result = np.where(mask == 1, image_array, gray_image)
    
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
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Segment the image using the U-Net model
        segmented_image = segment_image(image)

        # Display the segmented result
        st.write("Segmented Image:")
        show_image(segmented_image)

if __name__ == "__main__":
    main()
