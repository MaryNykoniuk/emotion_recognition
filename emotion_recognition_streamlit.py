import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import dlib
import joblib
from feature_extraction import find_features

emotions = ['afraid', 'angry', 'disgusted', 'happy', 'neutral', 'sad', 'surprised']
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
emotion_detector = tf.keras.models.load_model("my_model.h5")
scaler = joblib.load('scaler.save')


def process_image(cv2_img):
    image_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = detector(image_gray, 1)

    for face in faces:
        l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(cv2_img, (l, t), (r, b), (0, 255, 255), 2)
        landmarks = predictor(image_gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(cv2_img, (x, y), 1, (0, 255, 0), -1)

        features = find_features(landmarks)
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = emotion_detector.predict(features_scaled.reshape(1, -1))
        predicted_class = np.argmax(prediction)
        cv2.putText(cv2_img, emotions[predicted_class], (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return cv2_img


def main():
    st.set_page_config(page_title="Emotion Recognition App")
    st.title("Emotion Recognition App")
    enable = st.checkbox("Enable camera")
    img_file_buffer = st.camera_input("Take a picture", disabled=not enable)

    if img_file_buffer is not None:
        st.write("Wait a few seconds :sunglasses:")
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        processed_img = process_image(cv2_img)
        st.image(processed_img, channels="BGR")


if __name__ == "__main__":
    main()
