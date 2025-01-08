from tensorflow.keras.utils import register_keras_serializable
import tensorflow as tf
import os
import requests
from PIL import Image
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
import gdown

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

# Set Streamlit page configuration
st.set_page_config(page_title="Watch Segmentation with UNet", page_icon="⌚", layout="wide")

# Custom CSS for the app
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
tab = st.sidebar.radio("Select Tab", ["Watch Segmentation", "Feedbacks", "About"])

# Define model path and Google Drive file ID
model_path = "/tmp/unet.keras"
google_drive_file_id = "11MstxV8kt1fEHiLtnAe38kjgU7ru9xik"  # Replace with your actual file ID

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

# Watch Segmentation Tab
if tab == "Watch Segmentation":
    st.markdown('<div class="title">⌚ Watch Segmentation with UNet ⌚</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload an image of a watch, and the UNet model will segment it for you.</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)

        # Check model input shape
        model_input_shape = model.input_shape
        st.write(f"Model expects input shape: {model_input_shape}")

        # Preprocess the uploaded image
        try:
            img_resized = image.resize((model_input_shape[1], model_input_shape[2]))  # Resize to model input size
            img_rgb = img_resized.convert("RGB")  # Ensure 3 channels (RGB)
            img_array = img_to_array(img_rgb) / 255.0  # Normalize to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Check the prepared input shape
            st.write(f"Prepared input shape for the model: {img_array.shape}")

            # Run inference using the UNet model
            st.write("Running segmentation... Please wait.")
            prediction = model.predict(img_array)
            segmented_image = np.squeeze(prediction)  # Remove batch dimension

            # Convert segmentation mask to binary image
            mask = (segmented_image > 0.5).astype(np.uint8)  # Binarize mask

            # Create grayscale background and colorize the watch
            img_array_resized = np.array(img_resized)
            grayscale_background = np.dot(img_array_resized[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale

            # Apply the mask to the original image
            overlay = img_array_resized.copy()
            overlay[mask == 0] = grayscale_background[mask == 0][:, None]  # Apply grayscale to background

            # Display images side by side
            col1, col2 = st.columns(2)
            with col1:
                st.image(overlay, caption="Segmented Watch", use_column_width=True)
            with col2:
                st.image(img_resized, caption="Original Image", use_column_width=True)

            # Save and provide a download button for the segmented image
            output_image_path = "/tmp/segmented_watch.png"
            Image.fromarray(overlay).save(output_image_path)

            st.download_button(
                "Download Segmented Image",
                data=open(output_image_path, "rb"),
                file_name="segmented_watch.png",
                help="Click to download the segmented watch image."
            )

            # Feedback section
            st.subheader("Feedback")
            feedback = st.text_area("Please provide your feedback here:")
            if st.button("Submit Feedback"):
                if feedback:
                    st.success("Thank you for your feedback!")
                    # You can add logic here to save the feedback to a file or send it via email
                    # Example: save feedback to a text file
                    with open("feedback.txt", "a") as f:
                        f.write(feedback + "\n")
                else:
                    st.warning("Please enter some feedback before submitting.")
        except Exception as e:
            st.error(f"Error during preprocessing or prediction: {e}")

# Feedback Tab
elif tab == "Feedbacks":
    st.markdown('<div class="title">User Feedbacks</div>', unsafe_allow_html=True)

    # Read feedbacks from the file
    if os.path.exists("feedback.txt"):
        with open("feedback.txt", "r") as f:
            feedbacks = f.readlines()

        if feedbacks:
            st.write("Here are the feedbacks:")
            for feedback in feedbacks:
                st.write(f"- {feedback}")
        else:
            st.write("No feedbacks yet.")
    else:
        st.write("No feedbacks file found.")

# About Tab
elif tab == "About":
    st.markdown('<div class="title">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="description">
        This project uses a UNet deep learning model to segment watches from images. The model is trained with a custom dataset using 
        advanced loss functions such as IoU loss and metrics like IoU coefficient to achieve high accuracy.
        </div>
    """, unsafe_allow_html=True)
    st.write("""
    ### Features:
    - Upload an image for segmentation of watch.
    - Download the segmented image.
    - Built with Streamlit and TensorFlow.

    ### Contact:
    For inquiries or issues, please contact [Monica](mailto:evangelinemonica18@gmail.com).
    """)
