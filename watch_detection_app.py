import streamlit as st

# Customize CSS for extra styling
st.markdown(
    """
    <style>
    .main-header { font-size: 28px; color: #008080; font-weight: bold; }
    .sub-header { font-size: 20px; color: #2D2D2D; }
    .info-box { background-color: #66CCCC; padding: 20px; border-radius: 10px; color: #2D2D2D; }
    .footer { text-align: center; font-size: 14px; color: #666666; margin-top: 50px; }
    .btn-primary { background-color: #008080 !important; color: #FFFFFF !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown('<h1 class="main-header">AI-Powered Watch Detection System</h1>', unsafe_allow_html=True)

# Introduction
st.markdown('<p class="sub-header">Upload an image to detect and identify the watch in it.</p>', unsafe_allow_html=True)

# Step 1: Upload Image
st.header("Step 1: Upload Image")
uploaded_file = st.file_uploader("Upload an image of the watch you want to detect:", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# Step 2: Image Pre-processing
if uploaded_file:
    st.header("Step 2: Image Pre-processing")
    st.markdown(
        '<div class="info-box">The image will be pre-processed to enhance detection accuracy. This includes resizing, contrast adjustment, and noise reduction.</div>',
        unsafe_allow_html=True
    )

# Step 3: Watch Detection
if uploaded_file:
    st.header("Step 3: Watch Detection")
    if st.button("Detect Watch", key="detect", help="Click to run the detection algorithm"):
        # Placeholder for detection logic
        st.success("Watch detected successfully!")
        st.image("path_to_detected_watch_image.jpg", caption="Detection Result with Bounding Box")

# Step 4: Display Results with Bounding Boxes and Brand Details
if uploaded_file:
    st.header("Step 4: Display Results")
    with st.expander("View Detection Details"):
        st.write("### Bounding Box Coordinates")
        st.write("- Top-left: (x1, y1)")
        st.write("- Bottom-right: (x2, y2)")
        st.write("### Detected Brand: Omega")
        st.write("Confidence: 95%")

# Step 5: View Results
if uploaded_file:
    st.header("Step 5: View Results")
    st.markdown(
        '<div class="info-box">The results include detected brand details and bounding boxes around the detected watch. This will assist in further analysis and recognition.</div>',
        unsafe_allow_html=True
    )

# Step 6: Provide Feedback
st.header("Step 6: Provide Feedback")
st.text_area("Leave your feedback on the detection result to help us improve:", placeholder="Enter feedback here...")
if st.button("Submit Feedback", key="feedback"):
    st.success("Thank you for your feedback!")

# Footer
st.markdown('<div class="footer">Developed by Evangeline Monica (S2123599)</div>', unsafe_allow_html=True)
