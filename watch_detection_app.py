from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf
import os
import gdown
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array

# Register custom loss function
smooth = 1e-15

@register_keras_serializable()
def iou_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)

# Set Streamlit page configuration for a better UI
st.set_page_config(page_title="Watch Segmentation with UNet", page_icon="⌚", layout="wide")

# Custom CSS for the app's appearance
st.markdown("""
    <style>
        .title {
            color: #4a90e2;
            font-size: 36px;
            font-family: 'Roboto', sans-serif;
            text-align: center;
            margin-bottom: 20px;
        }
        .description {
            color: #444;
            font-size: 18px;
            font-family: 'Arial', sans-serif;
            text-align: center;
            margin-bottom: 40px;
        }
        .stButton button {
            background-color: #4a90e2;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stFileUploader label {
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
            padding: 12px 20px;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Watch Segmentation", "About"])

# Define model path and Google Drive file ID
model_path = "/tmp/unet.keras"
google_drive_file_id = "1YWxs3feor6QgdaJwRERY2yqcK4_TGCVK"  # Replace with your actual file ID

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    st.write("Downloading UNet model... Please wait.")
    try:
        gdown.download(f"https://drive.google.com/uc?id={google_drive_file_id}", model_path, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download the model: {e}")
        st.stop()

# Load the UNet model with custom objects
try:
    model = tf.keras.models.load_model(model_path, custom_objects={'iou_loss': iou_loss, 'iou_coef': iou_coef})
    st.success("UNet model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Page: Watch Segmentation
if page == "Watch Segmentation":
    st.markdown('<div class="title">⌚ Watch Segmentation with UNet ⌚</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload an image of a watch, and the UNet model will segment it for you.</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True, clamp=True)

        # Prepare the image for the UNet model
        img_resized = image.resize((512, 512))  # Resize to model input size used during training
        img_array = img_to_array(img_resized) / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run inference using UNet for segmentation
        st.write("Running segmentation... Please wait.")
        prediction = model.predict(img_array)
        segmented_image = np.squeeze(prediction)  # Remove batch dimension

        # Convert segmentation mask to a binary image and overlay it on the original
        mask = (segmented_image > 0.5).astype(np.uint8)  # Binarize mask
        overlay = np.array(image.resize((512, 512)))
        overlay[mask == 0] = [0, 0, 0]  # Make background black in the overlay

        # Display segmentation results
        st.image(overlay, caption="Segmented Watch", use_column_width=True)

        # Save and provide a download button for the segmented image
        output_image_path = "/tmp/segmented_watch.png"
        Image.fromarray(overlay).save(output_image_path)

        st.download_button(
            "Download Segmented Image",
            data=open(output_image_path, "rb"),
            file_name="segmented_watch.png",
            help="Click to download the segmented watch image."
        )

# Page: About
elif page == "About":
    st.markdown('<div class="title">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="description">
        This project uses a UNet deep learning model to segment watches from images. The model is trained with a custom dataset using 
        advanced loss functions such as IoU loss and metrics like IoU coefficient to achieve high accuracy.
        </div>
    """, unsafe_allow_html=True)
    st.write("""
    ### Features:
    - Upload an image of a watch for segmentation.
    - Download the segmented image.
    - Built with Streamlit and TensorFlow.

    ### Contact:
    For inquiries or issues, please contact [Monica](mailto:monica@example.com).
    """)
