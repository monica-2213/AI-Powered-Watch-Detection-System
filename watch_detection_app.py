import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

# Function to download the model from a URL
def download_model(url, model_path):
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    st.success("Model downloaded successfully!")

# URL of the pre-trained model (Replace this with your model's URL)
model_url = 'https://drive.google.com/uc?id=1-3SRZig9bvYvUYQmIVjO54l2oLlafpG4'  # Update this with your actual model URL
model_path = 'unet-non-aug.keras'  # Path to save the downloaded model

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    download_model(model_url, model_path)
    
# Define dice loss and dice coefficient functions if required by the model
# These functions should match the ones used in the training of the U-Net model
smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Load the pre-trained U-Net model
model = load_model(
    model_path,
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

def segment_image(uploaded_image):
    # Read and process the uploaded image
    original_image = np.array(uploaded_image)

    # Preprocess the image
    input_image = preprocess_image(original_image)

    # Make predictions
    predicted_mask = model.predict(input_image)

    # Handle binary segmentation
    predicted_mask = predicted_mask[0].squeeze()  # Remove batch and channel dimensions

    # Thresholding
    threshold = 0.1  # Adjust threshold as needed
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # Resize binary mask to match original image size if necessary
    binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

    # Create an image with the segmented area in color and the rest in grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_background = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

    # Combine the color-segmented area with the grayscale background
    color_segmented_image = np.where(binary_mask_resized[..., None] == 1, original_image, grayscale_background)

    return original_image, binary_mask_resized, color_segmented_image, predicted_mask

# Streamlit UI
st.title("AI Powered Watch Detection System")
st.write("Upload an image to detect and segment the watch.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to a format suitable for OpenCV
    image = Image.open(uploaded_file)
    original_image = np.array(image)

    # Display the original image
    st.image(original_image, caption="Uploaded Image", use_column_width=True)

    # Segment the image using the model
    original_image, binary_mask, color_segmented_image, predicted_mask = segment_image(original_image)

    # Display the results
    st.subheader("Predicted Segmentation Mask")
    st.image(binary_mask, caption="Predicted Mask", use_column_width=True, clamp=True, channels="GRAY")

    st.subheader("Segmented Area in Color with Grayscale Background")
    st.image(color_segmented_image, caption="Segmented Watch", use_column_width=True)

    # Optionally display the predicted probability map
    predicted_mask_scaled = (predicted_mask * 255).astype(np.uint8)
    st.subheader("Predicted Probability Map")
    st.image(predicted_mask_scaled, caption="Predicted Probability", use_column_width=True, clamp=True, channels="GRAY")
