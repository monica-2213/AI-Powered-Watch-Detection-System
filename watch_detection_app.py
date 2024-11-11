import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your trained U-Net model
model = load_model(
    '/content/drive/MyDrive/27_Oct_2024_Dataset_Creation/StratifiedDataset/newfiles_11Nov/unet-non-aug.keras',
    custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
)

def preprocess_image(image):
    # Resize to the expected input size (512, 512)
    image = cv2.resize(image, (512, 512))  # Note: (width, height) format
    # Normalize the image (if necessary)
    image = image / 255.0  # Scale pixel values to [0, 1]
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def segment_image(image):
    # Preprocess the image
    input_image = preprocess_image(image)

    # Make predictions
    predicted_mask = model.predict(input_image)

    # Debugging outputs
    print("Input Image Shape:", input_image.shape)
    print("Predicted Mask Shape:", predicted_mask.shape)
    print("Predicted Mask Unique Values:", np.unique(predicted_mask))

    # Handle binary segmentation
    predicted_mask = predicted_mask[0].squeeze()  # Remove batch and channel dimensions

    # Thresholding
    threshold = 0.1  # Adjust threshold as needed
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    return binary_mask, predicted_mask

def display_results(original_image, binary_mask, predicted_mask):
    # Resize binary mask to match original image size if necessary
    binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

    # Create an image with the segmented area in color and the rest in grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_background = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # Combine the color-segmented area with the grayscale background
    color_segmented_image = np.where(binary_mask_resized[..., None] == 1, original_image, grayscale_background)

    # Display the results using Streamlit
    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

    st.image(binary_mask_resized, caption="Predicted Segmentation Mask", use_column_width=True)

    predicted_mask_scaled = (predicted_mask * 255).astype(np.uint8)
    st.image(predicted_mask_scaled, caption="Predicted Probability Map", use_column_width=True, channels="BGR")

    st.image(cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2RGB), caption="Segmented Area in Color with Grayscale Background", use_column_width=True)

# Streamlit file uploader
st.title("Watch Image Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert the uploaded file to a NumPy array
    file_bytes = uploaded_file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    original_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Segment the image
    binary_mask, predicted_mask = segment_image(original_image)

    # Display the results
    display_results(original_image, binary_mask, predicted_mask)
