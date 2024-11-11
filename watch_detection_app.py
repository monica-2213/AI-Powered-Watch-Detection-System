import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
import os
model_path = 'unet-non-aug.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print(f"Model file {model_path} not found!")

def process_image(image):
    # Convert the image to RGB
    image_rgb = image.convert('RGB')
    
    # Resize to 512x512 for model input
    image_resized = image_rgb.resize((512, 512))
    
    # Convert image to numpy array and normalize it
    image_array = np.array(image_resized).astype(np.float32) / 255.0  # Normalize the image
    
    # Add batch and channel dimensions
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension if necessary
    
    # Perform prediction
    prediction = model.predict(image_array)
    
    return prediction[0], image_rgb.size  # Return both mask and original size
    

def segment_image(image):
    # Convert the original image to grayscale (using PIL)
    gray_image = image.convert('L')
    
    # Process the image through the model to get the segmentation mask
    mask, original_size = process_image(image)
    
    # Convert the mask to binary (assumes binary segmentation)
    mask_binary = mask > 0.5
    
    # Resize the mask back to the original image size
    mask_resized = Image.fromarray((mask_binary.astype(np.uint8) * 255).astype(np.uint8))
    mask_resized = mask_resized.resize(original_size, Image.BICUBIC)
    
    # Convert original image to numpy array for manipulation
    image_array = np.array(image)
    gray_image_array = np.array(gray_image)
    
    # Create an RGB output where the watch is in color and background is grayscale
    segmented_image = image_array.copy()
    segmented_image[~mask_binary] = np.stack([gray_image_array]*3, axis=-1)[~mask_binary]
    
    # Convert numpy array back to image
    segmented_image_pil = Image.fromarray(segmented_image)
    return segmented_image_pil

def display_segmented_image(segmented_image):
    # Display the segmented image
    st.image(segmented_image, caption="Segmented Image (Watch in Color, Background in Grayscale)", use_column_width=True)

def main():
    st.title('Watch Image Segmentation')
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Segment the image
        segmented_image = segment_image(image)

        # Display the segmented image
        display_segmented_image(segmented_image)

if __name__ == "__main__":
    main()
