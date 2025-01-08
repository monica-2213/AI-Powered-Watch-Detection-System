# Use a smaller base image for Python 3.10
FROM python:3.10-slim

# Expose the port that your app will run on
EXPOSE 8080

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files to avoid unnecessary overhead
COPY requirements.txt ./

# Install dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . ./ 

# If you're using a specific model file, use a multi-stage build to reduce size
# or make sure your model is downloaded after the initial setup
# RUN gdown download...

# Streamlit app entry point
ENTRYPOINT ["streamlit", "run", "watch_detection_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
