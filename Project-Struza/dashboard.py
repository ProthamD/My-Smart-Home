import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import time

# Streamlit config
st.set_page_config(page_title="Face Recognition System", layout="wide")
st.title("Face Recognition System")

# Create directory for authorized faces if it doesn't exist
KNOWN_FACES_DIR = "authorized_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Global variables for face recognition
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []

# Load known faces function
def load_known_faces():
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
    KNOWN_FACE_ENCODINGS = []
    KNOWN_FACE_NAMES = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        return
    
    st.info(f"Loading faces from {KNOWN_FACES_DIR}...")
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    KNOWN_FACE_ENCODINGS.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    # Remove digits from name if needed
                    name = ''.join([i for i in name if not i.isdigit()])
                    KNOWN_FACE_NAMES.append(name)
                    st.success(f"Loaded face: {name} from {filename}")
                else:
                    st.warning(f"No face found in {filename}")
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
    
    st.info(f"Total faces loaded: {len(KNOWN_FACE_ENCODINGS)}")

# UI layout
status_placeholder = st.empty()
camera_feed = st.empty()
col1, col2 = st.columns(2)

with col1:
    camera_button = st.button("Start Camera")

with col2:
    if st.button("Reload Known Faces"):
        load_known_faces()

# Initial load of faces
load_known_faces()

# Face management section
with st.expander("Manage Authorized Faces"):
    st.subheader("Current Authorized Faces")
    
    if not KNOWN_FACE_NAMES:
        st.info("No authorized faces found. Please add some faces.")
    else:
        st.success(f"Loaded {len(KNOWN_FACE_NAMES)} authorized faces:")
        for name in sorted(set(KNOWN_FACE_NAMES)):
            st.text(f"• {name}")
    
    st.subheader("Add New Authorized Face")
    uploaded_file = st.file_uploader("Upload a clear frontal face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image_array = np.array(image)
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) > 0:
            st.success(f"Found {len(face_locations)} face(s) in the image!")
            person_name = st.text_input("Enter the person's name")
            if st.button("Save as Authorized Face") and person_name:
                clean_name = ''.join(e for e in person_name if e.isalnum())
                
                # Create unique filename
                index = 1
                while True:
                    new_filename = f"{clean_name}{index}.jpg"
                    new_path = os.path.join(KNOWN_FACES_DIR, new_filename)
                    if not os.path.exists(new_path):
                        break
                    index += 1
                
                # Save the image
                image.save(new_path)
                st.success(f"Saved as {new_filename}")
                
                # Reload faces
                load_known_faces()
        else:
            st.error("No face detected in the image. Please upload a clear frontal face photo.")

# Camera function with face recognition
def run_camera():
    # Try to access the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_placeholder.error("Error: Could not open camera. Please check if camera is connected.")
        return
        
    status_placeholder.success("Camera is active. Press 'Stop Camera' when finished.")
    stop_button = st.button("Stop Camera")
    
    while cap.isOpened() and not stop_button:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to capture frame from camera")
            break
            
        # Convert the BGR frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Process each face found
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(KNOWN_FACE_ENCODINGS, face_encoding)
            name = "Unknown"
            
            # Find the best match
            if len(KNOWN_FACE_ENCODINGS) > 0:
                face_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = KNOWN_FACE_NAMES[best_match_index]
            
            # Set color based on recognition (green for known, red for unknown)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Display the resulting frame in the Streamlit app
        camera_feed.image(frame, channels="BGR", use_container_width=True)
        
        # Check if stop button pressed
        if stop_button:
            break
            
        # Add small delay to reduce CPU usage
        time.sleep(0.05)
    
    # Release the camera
    cap.release()
    status_placeholder.info("Camera stopped")
    camera_feed.empty()

# Run camera when button is pressed
if camera_button:
    run_camera()