import streamlit as st
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tempfile
import os

class FaceRecognition:
    """
    A class for performing face recognition using InceptionResNetV2 and MTCNN.
    """
    def __init__(self):
        """Initializes the FaceRecognition class with necessary attributes."""
        self.detector = None
        self.model = None
        self.stored_embeddings = None
        self.stored_labels = None

    def load_model(self):
        """Loads the InceptionResNetV2 model if not already loaded."""
        if self.model is None:
            base_model = InceptionResNetV2(weights="imagenet", include_top=False, pooling="avg")
            self.model = Model(inputs=base_model.input, outputs=base_model.output)

    def load_embeddings(self):
        """Loads stored face embeddings and labels from a pickle file."""
        if self.stored_embeddings is None or self.stored_labels is None:
            with open("face_embeddings.pkl", "rb") as f:
                self.stored_embeddings, self.stored_labels = pickle.load(f)

    def load_detector(self):
        """Loads the MTCNN face detector if not already loaded."""
        if self.detector is None:
            self.detector = MTCNN()

    def find_similar_faces(self, embedding, threshold=0.9):
        """
        Finds the most similar face from stored embeddings.
        
        Parameters:
        embedding (numpy.ndarray): The embedding of the detected face.
        threshold (float): The similarity threshold for matching faces.
        
        Returns:
        str: Matched person's name or None.
        float: Similarity score or None.
        """
        self.load_embeddings()
        similarities = cosine_similarity(embedding.reshape(1, -1), self.stored_embeddings)
        similar_indices = np.where(similarities[0] > threshold)[0]
        
        if len(similar_indices) == 0:
            return None, None
        
        best_idx = similar_indices[np.argmax(similarities[0][similar_indices])]
        return self.stored_labels[best_idx], similarities[0][best_idx]

    def process_frame(self, frame):
        """
        Detects faces in a frame, extracts embeddings, and identifies them.
        
        Parameters:
        frame (numpy.ndarray): The input frame/image.
        
        Returns:
        numpy.ndarray: Processed frame with drawn face boxes and labels.
        """
        self.load_detector()
        self.load_model()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame)
        
        for face in faces:
            confidence = face['confidence']
            if confidence < 0.9:
                continue
            
            x, y, w, h = face['box']
            x, y = max(x, 0), max(y, 0)
            
            face_img = rgb_frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (299, 299))
            face_array = np.expand_dims(face_img, axis=0)
            face_array = preprocess_input(face_array)
            
            embedding = self.model.predict(face_array, verbose=0)
            match_name, similarity = self.find_similar_faces(embedding)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if match_name is not None:
                text = f"{match_name}: {similarity:.2f}"
                cv2.putText(frame, text, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

def main():
    """Runs the Streamlit face recognition application."""
    st.title("Face Recognition System")
    
    # Initialize face recognition system
    if 'face_rec' not in st.session_state:
        st.session_state.face_rec = FaceRecognition()

    # Sidebar options
    st.sidebar.title("Options")
    detection_mode = st.sidebar.radio(
        "Choose Detection Mode",
        ["Upload Image", "Upload Video", "Webcam"]
    )

    if detection_mode == "Upload Image":
        image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None:
            try:
                # Convert uploaded file to image
                image = Image.open(image_file)
                img_array = np.array(image)
                
                # Process image
                processed_image = st.session_state.face_rec.process_frame(img_array)
                
                # Display results
                st.image(processed_image, caption='Processed Image', use_column_width=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif detection_mode == "Upload Video":
        video_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        
        if video_file is not None:
            try:
                # Save uploaded file temporarily
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                tfile.close()
                
                # Open video file
                cap = cv2.VideoCapture(tfile.name)
                
                # Video player placeholder
                video_placeholder = st.empty()
                
                # Process video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = st.session_state.face_rec.process_frame(frame)
                    video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                
                cap.release()
                os.unlink(tfile.name)
            except Exception as e:
                st.error(f"Error processing video: {e}")

    elif detection_mode == "Webcam":
        """Handles real-time face detection using a webcam."""
        run = st.checkbox('Start Webcam')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        while run:
            try:
                _, frame = camera.read()
                if frame is not None:
                    processed_frame = st.session_state.face_rec.process_frame(frame)
                    FRAME_WINDOW.image(processed_frame, channels="BGR")
            except Exception as e:
                st.error(f"Error capturing webcam: {e}")
                break
        
        if not run:
            camera.release()
            st.write('Webcam stopped')

if __name__ == "__main__":
    main()
