import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io
import matplotlib.pyplot as plt

# Load the U-Net model locally
def load_model_from_local():
    model = load_model('unet-non-aug.keras', custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    return model

# Preprocess the uploaded image
def preprocess_image(image):
    # Resize to the expected input size (512, 512)
    image = cv2.resize(image, (512, 512))  # Note: (width, height) format
    # Normalize the image
    image = image / 255.0  # Scale pixel values to [0, 1]
    # Expand dimensions to match model input shape
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Predict the mask using the model
def predict_mask(model, input_image):
    predicted_mask = model.predict(input_image)
    return predicted_mask

# Display segmented output
def display_output(original_image, binary_mask_resized, predicted_mask):
    # Create an image with the segmented area in color and the rest in grayscale
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    grayscale_background = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
    
    color_segmented_image = np.where(binary_mask_resized[..., None] == 1, original_image, grayscale_background)
    
    # Plot the results
    fig, axes = plt.subplots(1, 4, figsize=(20, 8))

    # Original image
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Predicted segmentation mask
    axes[1].imshow(binary_mask_resized, cmap='gray')
    axes[1].set_title('Predicted Segmentation Mask')
    axes[1].axis('off')

    # Predicted probability map (optional)
    predicted_mask_scaled = (predicted_mask[0] * 255).astype(np.uint8)
    axes[2].imshow(predicted_mask_scaled, cmap='jet')
    axes[2].set_title('Predicted Probability Map')
    axes[2].axis('off')

    # Segmented area in color with grayscale background
    axes[3].imshow(cv2.cvtColor(color_segmented_image, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Segmented Area (Watch) in Color')
    axes[3].axis('off')

    # Show plot in Streamlit
    st.pyplot(fig)

# Streamlit UI setup
st.title("Watch Segmentation with U-Net")

st.markdown("""
    <style>
        .main {
            background-color: #F0F8FF;
            color: #1E90FF;
            font-family: Arial, sans-serif;
        }
        .upload-btn {
            background-color: #32CD32;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px;
        }
        .upload-btn:hover {
            background-color: #228B22;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("Upload your image:")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load the model locally
model = load_model_from_local()

if uploaded_file is not None:
    # Convert the uploaded image to a NumPy array
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Preprocess the image
    input_image = preprocess_image(image)

    # Make predictions
    predicted_mask = predict_mask(model, input_image)

    # Post-process the prediction
    predicted_mask = predicted_mask[0].squeeze()  # Remove batch and channel dimensions
    threshold = 0.1  # Adjust threshold as needed
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # Resize the mask to match the original image
    binary_mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))

    # Display the results
    display_output(image, binary_mask_resized, predicted_mask)

    st.success("Segmentation completed successfully!")
else:
    st.info("Please upload an image to begin segmentation.")
