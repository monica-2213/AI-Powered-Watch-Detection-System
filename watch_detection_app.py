import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import io
import tensorflow as tf

# Define Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Define Dice Coefficient
def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    
# Load your trained U-Net model
model = load_model(
    '/content/drive/MyDrive/27_Oct_2024_Dataset_Creation/StratifiedDataset/newfiles_11Nov/unet-non-aug.keras',
    custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
)

def preprocess_image(image):
    # Resize to the expected input size (512, 512)
    image = image.resize((512, 512))  # Resize using Pillow (width, height)
    # Normalize the image (if necessary)
    image = np.array(image) / 255.0  # Convert to numpy array and scale pixel values to [0, 1]
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
    binary_mask_resized = binary_mask  # Already has the same shape

    # Convert the original image to grayscale using Pillow
    grayscale_image = original_image.convert("L")
    grayscale_image = np.array(grayscale_image)

    # Create a grayscale background
    grayscale_background = np.stack([grayscale_image] * 3, axis=-1)

    # Combine the color-segmented area with the grayscale background
    color_segmented_image = np.where(binary_mask_resized[..., None] == 1, np.array(original_image), grayscale_background)

    # Display the results using Streamlit
    st.image(original_image, caption="Original Image", use_column_width=True)

    st.image(binary_mask_resized, caption="Predicted Segmentation Mask", use_column_width=True)

    predicted_mask_scaled = (predicted_mask * 255).astype(np.uint8)
    st.image(predicted_mask_scaled, caption="Predicted Probability Map", use_column_width=True, channels="BGR")

    st.image(color_segmented_image, caption="Segmented Area in Color with Grayscale Background", use_column_width=True)

# Streamlit file uploader
st.title("Watch Image Segmentation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image using Pillow
    image = Image.open(uploaded_file)

    # Segment the image
    binary_mask, predicted_mask = segment_image(image)

    # Display the results
    display_results(image, binary_mask, predicted_mask)
