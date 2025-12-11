# Brain_Tumor_Detection

  A deep-learning based system to classify MRI brain images as Tumor or No Tumor using a Convolutional Neural Network (CNN).

**Overview**

  Brain tumor detection is one of the most critical tasks in the field of medical image analysis.
  This project uses a custom CNN (Convolutional Neural Network) model to automatically analyze MRI images and classify them for tumor presence.

**Problem Statement**

  Early detection of brain tumors is essential for successful treatment, but manual MRI analysis:

    Is time-consuming

    Requires specialized radiologists

    Is prone to human error

    The goal is to build a fast, accurate, and automated tumor-classification model.

**Motivation**

  Deep learning has proved to be highly effective in medical image analysis.
  By developing a CNN-based detector, this project aims to assist:

    Radiologists

    Healthcare professionals

    AI-powered diagnostic tools

    This system can help improve diagnostic accuracy and reduce workload.

**Literature Review**

  Many studies have shown the power of CNNs for MRI classification:

    CNN models outperform classical machine learning methods like SVM and KNN

    Transfer Learning models (VGG, ResNet) achieve high accuracy on medical datasets

    Data augmentation greatly enhances model generalization

    These studies validate the use of CNNs for brain tumor identification.

**Methodology**

  1. Dataset

          Common datasets used:

          Figshare Brain MRI Dataset

          Kaggle Brain Tumor MRI Dataset

          Classes:

          Tumor

          No Tumor

  2. Preprocessing

         Resize all images (e.g., 128×128 or 224×224)
    
         Convert to RGB (if required)
    
          Normalize pixel values (0–1)
    
         Apply data augmentation:
  
         Flip
  
         Rotation
  
         Zoom
  
         Noise addition
  
         Brightness variation
  3. CNN Architecture

          Typical architecture used:
    
            Input (128x128x3)
            ↓
            Conv2D → ReLU
            ↓
            MaxPooling
            ↓
            Conv2D → ReLU
            ↓
            MaxPooling
            ↓
            Flatten
            ↓
            Dense → ReLU
            ↓
            Dropout
            ↓
            Dense (2 neurons → Softmax)

  4. Model Training

          Optimizer: Adam
          
          Loss: Binary/Categorical Crossentropy
          
          Metrics: Accuracy
          
          Epochs: 30–50
          
          The model is trained using 80% of the dataset with validation on the remaining 20%.

**System Requirements**

  Hardware

      i5 or higher

      8–16 GB RAM

      NVIDIA GPU recommended for training

  Software

      Python 3.7+

      TensorFlow / Keras

      NumPy
      
      OpenCV
  
      Matplotlib

      Scikit-learn

**Example – Predicting Tumor from MRI**
    
    img = cv2.imread("test_mri.jpg")
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    result = "Tumor" if np.argmax(prediction) == 1 else "No Tumor"

    print("Prediction:", result)


    Output:
    Prediction: Tumor

**Results**

  The model achieves:
  
      High accuracy on both training and test data
      
      Good generalization due to augmentation
      
      Reliable performance even on noisy MRI scans
      
      Performance metrics include:
      
      Accuracy
      
      Loss
      
      Confusion Matrix
      
      Precision / Recall

**Future Scope**

  Add more MRI datasets for generalization
  
  Deploy model using Flask / Streamlit
  
  Integrate into hospital systems
  
  Convert model to ONNX / TensorFlow Lite for faster inference
  
  Implement Grad-CAM for tumor visualization

**References**

  Deep Learning by Ian Goodfellow
  
  Research papers on CNN in medical imaging
  
  Kaggle & Figshare MRI datasets
  
  CNN literature on tumor classification
