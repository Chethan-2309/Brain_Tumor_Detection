import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image # --- ADDED --- To handle image files correctly

# --- ADDED --- Cache the model loading to improve performance
@st.cache_resource
def load_trained_model():
    """Loads the trained CNN model from the .keras file."""
    model = load_model('E:/Brain Tumor Detection/frontflask/my_model.keras')
    return model

# Load the model
model = load_trained_model()

# Set the title of the app
st.title("Brain Tumor Detection System 🧠")

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Add a button to trigger the prediction
    if st.button("Detect Tumor"):
        # --- ADDED --- Show a spinner while the model is making a prediction
        with st.spinner("Analyzing the image..."):
            
            # --- CHANGED --- Use Pillow to open and preprocess the image
            image = Image.open(uploaded_file)
            image = image.convert('L')  # Convert to grayscale
            image = image.resize((224, 224)) # Resize to match model's expected input
            
            # Convert the image to a numpy array and normalize it
            img_array = np.array(image)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
            img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension for grayscale

            # Make a prediction
            prediction = model.predict(img_array)

            # Display the result
            st.write("")
            if prediction[0][0] > 0.5:
                st.error("Prediction: **Tumor Detected** ")
            else:
                st.success("Prediction: **No Tumor Detected** ")