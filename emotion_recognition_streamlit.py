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


def main():
    st.set_page_config(page_title="Emotion Recognition App")
    st.title("Emotion Recognition App")

    show_landmarks = st.checkbox("Show landmarks on face")

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Video Capture Ended")
            break
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(image_gray, 1)
        for face in faces:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 2)
            landmarks = predictor(image_gray, face)

            if show_landmarks:
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            features = find_features(landmarks)
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = emotion_detector.predict(features_scaled.reshape(1, -1))
            predicted_class = np.argmax(prediction)

            cv2.putText(frame, emotions[predicted_class], (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        frame_placeholder.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
