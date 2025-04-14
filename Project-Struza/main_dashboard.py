import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import time
import datetime
import threading

# Streamlit config
st.set_page_config(page_title="Face Recognition Security System", layout="wide")
st.title("Face Recognition Security System")

# Create directories if they don't exist
KNOWN_FACES_DIR = "authorized_faces"
INTRUSION_DIR = "intrusion_evidence"
for directory in [KNOWN_FACES_DIR, INTRUSION_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Global variables
KNOWN_FACE_ENCODINGS = []
KNOWN_FACE_NAMES = []
last_intrusion_time = None
intrusion_cooldown = 30  # seconds between alerts

# App configuration sidebar
with st.sidebar:
    st.header("Security Settings")
    
    # Security mode settings
    security_mode = st.radio("Security Mode:", 
                            ("Monitor Only", "Active Alerting"),
                            index=0)
    
    # Alert settings
    st.subheader("Alert Settings")
    save_evidence = st.checkbox("Save Intrusion Evidence", value=True)
    
    # Recognition settings - kept simple to avoid performance issues
    st.subheader("Recognition Settings")
    face_tolerance = st.slider("Recognition Tolerance", 
                              min_value=0.4, max_value=0.8, value=0.6, 
                              help="Lower value = stricter matching")

# Load known faces function
def load_known_faces():
    global KNOWN_FACE_ENCODINGS, KNOWN_FACE_NAMES
    KNOWN_FACE_ENCODINGS = []
    KNOWN_FACE_NAMES = []
    
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        return
    
    status_text = st.empty()
    status_text.info(f"Loading faces from {KNOWN_FACES_DIR}...")
    
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                
                if len(encodings) > 0:
                    KNOWN_FACE_ENCODINGS.append(encodings[0])
                    name = os.path.splitext(filename)[0]
                    name = ''.join([i for i in name if not i.isdigit()])
                    KNOWN_FACE_NAMES.append(name)
                else:
                    st.warning(f"No face found in {filename}")
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
    
    status_text.success(f"Loaded {len(KNOWN_FACE_ENCODINGS)} authorized faces")

# Function to save intrusion evidence
def save_intrusion_image(frame, name="Unknown"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.jpg"
    filepath = os.path.join(INTRUSION_DIR, filename)
    cv2.imwrite(filepath, frame)
    return filepath

# Main UI layout
status_placeholder = st.empty()
alert_placeholder = st.empty()
camera_feed = st.empty()

col1, col2, col3 = st.columns(3)
with col1:
    camera_button = st.button("Start Camera")
with col2:
    if st.button("Reload Known Faces"):
        load_known_faces()
with col3:
    clear_alerts = st.button("Clear Alerts")
    if clear_alerts:
        alert_placeholder.empty()

# Initial load of faces
load_known_faces()

# Face management section
with st.expander("Manage Authorized Faces"):
    st.subheader("Current Authorized Faces")
    
    if not KNOWN_FACE_NAMES:
        st.info("No authorized faces found. Please add some faces.")
    else:
        face_cols = st.columns(3)
        unique_names = sorted(set(KNOWN_FACE_NAMES))
        for i, name in enumerate(unique_names):
            with face_cols[i % 3]:
                st.text(f"• {name}")
    
    st.subheader("Add New Authorized Face")
    uploaded_file = st.file_uploader("Upload a clear frontal face image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image_array = np.array(image)
        face_locations = face_recognition.face_locations(image_array)
        
        if len(face_locations) > 0:
            # Draw rectangles around detected faces in preview
            preview_img = image_array.copy()
            for top, right, bottom, left in face_locations:
                cv2.rectangle(preview_img, (left, top), (right, bottom), (0, 255, 0), 2)
            
            st.image(preview_img, caption=f"Found {len(face_locations)} face(s)", use_container_width=True)
            
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

# Show intrusion evidence
with st.expander("View Intrusion Evidence"):
    if os.path.exists(INTRUSION_DIR) and os.listdir(INTRUSION_DIR):
        st.subheader("Recent Intrusions")
        
        # Get list of evidence files sorted by date (newest first)
        evidence_files = sorted(
            [f for f in os.listdir(INTRUSION_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda x: os.path.getmtime(os.path.join(INTRUSION_DIR, x)),
            reverse=True
        )
        
        # Show the evidence in a grid
        evidence_cols = st.columns(3)
        for i, img_file in enumerate(evidence_files[:9]):  # Show up to 9 recent images
            with evidence_cols[i % 3]:
                img_path = os.path.join(INTRUSION_DIR, img_file)
                timestamp = datetime.datetime.fromtimestamp(
                    os.path.getmtime(img_path)
                ).strftime("%Y-%m-%d %H:%M:%S")
                
                st.image(img_path, caption=f"{timestamp}", use_container_width=True)
                
                if st.button(f"Delete", key=f"del_{i}"):
                    try:
                        os.remove(img_path)
                        st.success(f"Deleted {img_file}")
                        time.sleep(1)
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {str(e)}")
        
        if st.button("Clear All Evidence"):
            try:
                for file in evidence_files:
                    os.remove(os.path.join(INTRUSION_DIR, file))
                st.success("All evidence files deleted")
                time.sleep(1)
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing evidence: {str(e)}")
    else:
        st.info("No intrusion evidence found")

# Camera function with stable face recognition and alerts
def run_camera():
    global last_intrusion_time
    
    # Set security mode based on selection
    SECURITY_MODE_ACTIVE = (security_mode == "Active Alerting")
    
    # Try to access the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_placeholder.error("Error: Could not open camera. Please check if camera is connected.")
        return
        
    status_placeholder.success(f"Camera is active. Mode: {security_mode}")
    stop_button = st.button("Stop Camera")
    
    # Initialize detector for smaller faces - makes detection more robust
    # Using HOG model which is faster than CNN but still accurate enough
    detection_model = "hog"  # alternatively "cnn" for better but slower detection
    
    while cap.isOpened() and not stop_button:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to capture frame from camera")
            break
        
        # Process at a consistent rate - every frame for stability
        # Convert the BGR frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Downsample frame for faster processing (optional)
        # small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
        
        # Use the original frame for better accuracy
        small_frame = rgb_frame
        
        # Find all face locations and encodings
        face_locations = face_recognition.face_locations(small_frame, model=detection_model)
        
        # Only compute encodings if faces are found
        face_encodings = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        # Track if any unauthorized faces are detected in this frame
        unauthorized_detected = False
        
        # Process each face found
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # If we used downsampling, scale coordinates back up
            # top *= 2
            # right *= 2
            # bottom *= 2
            # left *= 2
            
            # Compare with known faces
            matches = face_recognition.compare_faces(
                KNOWN_FACE_ENCODINGS, 
                face_encoding, 
                tolerance=face_tolerance
            )
            name = "Unknown"
            
            # Find the best match
            if len(KNOWN_FACE_ENCODINGS) > 0:
                face_distances = face_recognition.face_distance(KNOWN_FACE_ENCODINGS, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = KNOWN_FACE_NAMES[best_match_index]
            
            # Set color based on recognition (green for known, red for unknown)
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                unauthorized_detected = True
            else:
                color = (0, 255, 0)  # Green for known
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Handle unauthorized face detection
        current_time = time.time()
        cooldown_passed = (last_intrusion_time is None or 
                          (current_time - last_intrusion_time) > intrusion_cooldown)
        
        if unauthorized_detected and SECURITY_MODE_ACTIVE and cooldown_passed:
            # Update last intrusion time
            last_intrusion_time = current_time
            
            # Save evidence if enabled
            if save_evidence:
                evidence_path = save_intrusion_image(frame)
                st.warning(f"Intrusion evidence saved: {os.path.basename(evidence_path)}")
            
            # Display alert
            alert_placeholder.error("⚠️ ALERT: Unauthorized person detected!")
        
        # Add security mode indicator to the frame
        mode_text = f"Mode: {'ACTIVE ALERTING' if SECURITY_MODE_ACTIVE else 'Monitor Only'}"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 255) if SECURITY_MODE_ACTIVE else (255, 0, 0), 2)
        
        # Display the resulting frame in the Streamlit app
        camera_feed.image(frame, channels="BGR", use_container_width=True)
        
        if stop_button:
            break
            

    
    # Release the camera
    cap.release()
    status_placeholder.info("Camera stopped")
    camera_feed.empty()

# Run camera when button is pressed
if camera_button:
    run_camera()