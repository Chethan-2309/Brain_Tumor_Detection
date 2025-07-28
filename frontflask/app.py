import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'static/uploads/'
EXAMPLE_FOLDER = 'static/example_images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Both Models
try:
    tumor_model = load_model('my_model.keras')
    gatekeeper_model = load_model('gatekeeper_model.keras')
except Exception as e:
    print(f"Error loading models: {e}")
    tumor_model, gatekeeper_model = None, None

# Helper function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# --- NEW: Route for the Dashboard ---
@app.route('/')
def dashboard():
    # List example images from the directory
    example_images = [os.path.join(EXAMPLE_FOLDER, f) for f in os.listdir(EXAMPLE_FOLDER) if f.endswith(('jpg', 'png', 'jpeg'))]
    return render_template('dashboard.html', example_images=example_images)

# --- NEW: Route for the Detector Page ---
@app.route('/detector')
def detector():
    return render_template('detector.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if not all([tumor_model, gatekeeper_model]):
        return "Server error: Models are not loaded.", 500

    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('detector.html', error_message="No file selected. Please upload an image.")
    
    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # 1. Gatekeeper Check
    gatekeeper_image = preprocess_image(filepath)
    is_mri_pred = gatekeeper_model.predict(gatekeeper_image)[0][0]

    if is_mri_pred > 0.5:
        return render_template('detector.html',
                               error_message="Invalid Image: Please upload a brain MRI scan.",
                               image_path=filepath)

    # 2. Tumor Detection
    prediction = tumor_model.predict(gatekeeper_image)[0][0]

    # Initialize variables
    suggestions, precautions, search_url, image_search_url = None, None, None, None

    if prediction > 0.5:
        prediction_text = "Tumor Detected"
        prediction_class = "error"
        suggestions = [
            "Consult a neuro-oncologist immediately for a detailed diagnosis.",
            "Undergo further imaging tests like MRI with contrast, CT scan, or PET scan.",
            "Discuss treatment options such as surgery, radiation therapy, or chemotherapy."
        ]
        search_url = "https://www.google.com/search?q=brain+tumor+treatment+options"
        image_search_url = "https://www.google.com/search?tbm=isch&q=brain+tumor+mri+scan"
    else:
        prediction_text = "No Tumor Detected"
        prediction_class = "success"
        precautions = [
            "Maintain a healthy lifestyle with a balanced diet and regular exercise.",
            "Avoid exposure to radiation and harmful chemicals.",
            "Get regular health check-ups to monitor your overall well-being."
        ]

    return render_template('detector.html',
                           prediction_text=prediction_text,
                           prediction_class=prediction_class,
                           image_path=filepath,
                           suggestions=suggestions,
                           precautions=precautions,
                           search_url=search_url,
                           image_search_url=image_search_url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)