# emotion_recognition

# Emotion Detection from Facial Landmarks

This project is focused on detecting human emotions based on facial landmark detection and feature extraction. The goal is to extract meaningful features (such as distances and angles between facial landmarks) and use them to classify different emotions (afraid, angry, disgusted, happy, neutral, sad, surprised).

## Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Installation](#installation)
- [How It Works](#howitworks)

## Overview
The **Emotion Detection from Facial Landmarks** project utilizes computer vision techniques to detect emotions from facial images. The process involves detecting faces, extracting facial landmarks, calculating geometric features (such as distances and angles between landmarks), and using these features to classify emotions.

### Features:
- Detects faces in images using a pre-trained face detector
- Extracts 68 facial landmarks for each detected face
- Calculates geometric features (distances and angles) between key facial points
- Classifies emotions based on the extracted features

### Demo:
You can check out a demo of this project hosted on **Streamlit** (if applicable) at the following link:
[Live Demo on Streamlit](https://emotionrecognition-marynyk.streamlit.app/)

In this demo, you can open your webcamera, create image, and the system will detect the face, extract landmarks, compute geometric features, and predict the emotion.

## Installation

### Prerequisites:
To run this project, you will need the following libraries and tools installed:

- **Python 3.7** or above
- **[OpenCV](https://opencv.org/)** for image processing
- **[dlib](http://dlib.net/)** for facial landmark detection
- **[NumPy](https://numpy.org/)** for numerical computations
- **[Streamlit](https://streamlit.io/)** for the web interface
- **[scikit-learn](https://scikit-learn.org/stable/)** for machine learning models
- **[TensorFlow](https://www.tensorflow.org/)** for deep learning models
- **[Joblib](https://joblib.readthedocs.io/en/latest/)** for loading machine learning models

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/emotion-detection.git
    cd emotion-detection
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If you're running the project with a web interface, you can run the following command to start the Streamlit app:
    ```bash
    streamlit run emotion_recognition_streamlit.py
    ```

## How It Works

1. **Face Detection**: 
   The project uses the `dlib` frontal face detector to locate faces in an image.

2. **Facial Landmark Detection**: 
   A pre-trained model (`shape_predictor_68_face_landmarks.dat`) is used to detect 68 key facial landmarks, including points for the eyes, nose, mouth, and jawline.

3. **Feature Extraction**: 
   For each face, geometric features such as distances and angles between certain key landmarks (e.g., eyes, mouth, nose) are calculated. These features are critical for emotion classification.

4. **Emotion Classification**: 
   Based on the calculated features, a model classifies the emotion into categories like afraid, angry, disgusted, happy, neutral, sad, surprised.
