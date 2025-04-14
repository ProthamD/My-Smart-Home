import streamlit as st
import cv2
import numpy as np
import face_recognition
import threading
import time
import os
import json
import pickle
import base64
import io
from datetime import datetime
from PIL import Image

# Set up the Streamlit page
st.set_page_config(page_title="Smart Home Security Hub", layout="wide")
st.title("Smart Home Security Hub")

# Create necessary directories if they don't exist
os.makedirs("authorized_faces", exist_ok=True)
os.makedirs("security_logs", exist_ok=True)

# Initialize session state variables
if 'system_active' not in st.session_state:
    st.session_state.system_active = False
if 'simulation_mode' not in st.session_state:
    st.session_state.simulation_mode = True
if 'authorized_faces' not in st.session_state:
    st.session_state.authorized_faces = {}
if 'security_logs' not in st.session_state:
    st.session_state.security_logs = []
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = None
if 'last_motion_time' not in st.session_state:
    st.session_state.last_motion_time = 0
if 'motion_detected' not in st.session_state:
    st.session_state.motion_detected = False
if 'door_status' not in st.session_state:
    st.session_state.door_status = "CLOSED"
if 'system_status' not in st.session_state:
    st.session_state.system_status = "STANDBY"
if 'alert_level' not in st.session_state:
    st.session_state.alert_level = "info"
if 'alert_message' not in st.session_state:
    st.session_state.alert_message = ""
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'face_recognition_active' not in st.session_state:
    st.session_state.face_recognition_active = True
if 'motion_sensitivity' not in st.session_state:
    st.session_state.motion_sensitivity = 0.5

# Thread control
stop_threads = threading.Event()
lock = threading.RLock()

# Function to find available cameras
def get_available_cameras():
    available_cameras = []
    for i in range(5):  # Try first 5 camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except Exception:
            pass
    return available_cameras

# Load or create face encodings database
def load_face_database():
    db_path = "authorized_faces/face_db.pkl"
    if os.path.exists(db_path):
        try:
            with open(db_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.sidebar.error(f"Error loading face database: {e}")
            return {}
    return {}

# Add person to face database
def add_person_to_database(name, face_encoding):
    with lock:
        st.session_state.authorized_faces[name] = face_encoding
        # Save to disk
        with open("authorized_faces/face_db.pkl", 'wb') as f:
            pickle.dump(st.session_state.authorized_faces, f)
        
        # Log the action
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": f"Added authorized person: {name}",
            "level": "info"
        }
        st.session_state.security_logs.append(log_entry)

# Log security events
def log_security_event(event, level="info", image=None):
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "level": level
    }
    
    if image is not None and level in ["warning", "error"]:
        # Save the event image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"security_logs/event_{timestamp}.jpg"
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        log_entry["image_path"] = image_path
    
    with lock:
        st.session_state.security_logs.append(log_entry)
        # Keep only the last 100 logs in memory
        if len(st.session_state.security_logs) > 100:
            st.session_state.security_logs = st.session_state.security_logs[-100:]
        
        # Also save to JSON log file
        try:
            log_file = "security_logs/security_log.json"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # Remove image data before saving to JSON
            if "image_data" in log_entry:
                del log_entry["image_data"]
                
            logs.append(log_entry)
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error saving log: {e}")

# Process motion detection
def detect_motion(frame, prev_frame, threshold=25):
    if prev_frame is None:
        return False, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    
    # Compute absolute difference
    frame_diff = cv2.absdiff(gray, prev_gray)
    
    # Threshold the diff
    _, thresh = cv2.threshold(frame_diff, int(threshold), 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by contour area
    min_area = frame.shape[0] * frame.shape[1] * 0.001 * st.session_state.motion_sensitivity
    motion_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Draw motion contours on a copy of the frame
    motion_frame = frame.copy()
    cv2.drawContours(motion_frame, motion_contours, -1, (0, 255, 0), 2)
    
    return len(motion_contours) > 0, motion_frame

# Simulate door/window sensor
def simulate_sensors():
    while not stop_threads.is_set():
        # Simulate occasional door opening with 5% probability every 30 seconds
        if st.session_state.simulation_mode and time.time() % 30 < 1 and random.random() < 0.05:
            with lock:
                prev_status = st.session_state.door_status
                st.session_state.door_status = "OPEN"
                if prev_status == "CLOSED":
                    log_security_event("Door opened", "warning")
            # Door stays open for 10 seconds
            time.sleep(10)
            with lock:
                st.session_state.door_status = "CLOSED"
                log_security_event("Door closed", "info")
        
        time.sleep(1)

# Video processing function
def process_video_feed(camera_index):
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            log_security_event(f"Failed to open camera at index {camera_index}", "error")
            st.session_state.system_active = False
            st.session_state.system_status = "ERROR"
            return
        
        log_security_event("Security camera activated", "info")
        st.session_state.system_status = "ACTIVE"
        
        prev_frame = None
        frame_count = 0
        face_recognition_interval = 5  # Process every 5th frame for face recognition
        
        unauthorized_cooldown = 0  # Cooldown timer for unauthorized alerts
        
        while st.session_state.system_active and not stop_threads.is_set():
            ret, frame = cap.read()
            if not ret:
                log_security_event("Camera disconnected", "error")
                break
                
            # Convert BGR to RGB for streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Motion detection on every frame
            motion_detected, motion_frame = detect_motion(frame_rgb, prev_frame)
            prev_frame = frame_rgb.copy()
            
            # Update motion status with cooldown
            if motion_detected:
                with lock:
                    st.session_state.motion_detected = True
                    st.session_state.last_motion_time = time.time()
                    
                # Log motion only once per detection event (with 5 second cooldown)
                if time.time() - st.session_state.last_motion_time < 5:
                    log_security_event("Motion detected", "warning", motion_frame)
            elif time.time() - st.session_state.last_motion_time > 3:  # 3 second cooldown
                with lock:
                    st.session_state.motion_detected = False
            
            # Face recognition processing (every Nth frame)
            display_frame = motion_frame if motion_detected else frame_rgb
            
            if st.session_state.face_recognition_active and frame_count % face_recognition_interval == 0:
                # Resize for faster processing
                small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
                
                # Get face locations and encodings
                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)
                
                # Process each face
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Scale back to original size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Compare with known faces
                    matches = []
                    match_name = "Unknown"
                    
                    if st.session_state.authorized_faces:
                        for name, known_encoding in st.session_state.authorized_faces.items():
                            # Compare face encodings
                            match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
                            if match[0]:
                                match_name = name
                                break
                    
                    # Determine if this is an authorized person
                    if match_name == "Unknown":
                        # Unauthorized person
                        color = (255, 0, 0)  # Red
                        
                        # Log unauthorized access with cooldown
                        if unauthorized_cooldown == 0:
                            if st.session_state.motion_detected or st.session_state.door_status == "OPEN":
                                alert_msg = "⚠️ ALERT: Unauthorized person detected with sensor activity!"
                                log_security_event(alert_msg, "error", frame_rgb)
                                st.session_state.alert_level = "error"
                                st.session_state.alert_message = alert_msg
                            else:
                                log_security_event("Unauthorized person detected", "warning", frame_rgb)
                                st.session_state.alert_level = "warning"
                                st.session_state.alert_message = "⚠️ Unknown person detected"
                            
                            unauthorized_cooldown = 30  # Set cooldown for 30 frames
                    else:
                        # Authorized person
                        color = (0, 255, 0)  # Green
                        
                        # Log authorized entry
                        if st.session_state.motion_detected or st.session_state.door_status == "OPEN":
                            log_security_event(f"Authorized access by {match_name}", "info")
                            st.session_state.alert_level = "info"
                            st.session_state.alert_message = f"✓ Welcome, {match_name}"
                    
                    # Draw face rectangle and name
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, match_name, (left, top - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                
                # Decrease unauthorized cooldown
                if unauthorized_cooldown > 0:
                    unauthorized_cooldown -= 1
            
            # Update frame counter
            frame_count += 1
            
            # Update display frame every other frame
            if frame_count % 2 == 0:
                with lock:
                    st.session_state.current_frame = display_frame
            
            # Short delay to reduce CPU usage
            time.sleep(0.05)
    
    except Exception as e:
        log_security_event(f"Camera error: {str(e)}", "error")
        st.session_state.system_status = "ERROR"
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        st.session_state.system_active = False
        st.session_state.system_status = "INACTIVE"
        log_security_event("Security camera deactivated", "info")

# Function to capture and add a new authorized face
def capture_authorized_face(camera_index, person_name):
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error(f"Failed to open camera at index {camera_index}")
            return False
        
        st.info("Capturing face... Please look at the camera.")
        
        # Countdown
        for i in range(3, 0, -1):
            st.write(f"Capturing in {i}...")
            time.sleep(1)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            return False
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            st.error("No face detected. Please try again.")
            return False
        
        # Get face encoding
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        if not face_encodings:
            st.error("Could not encode face. Please try again.")
            return False
        
        # Save encoding to database
        add_person_to_database(person_name, face_encodings[0])
        
        # Save face image to directory
        image_path = f"authorized_faces/{person_name}.jpg"
        cv2.imwrite(image_path, frame)
        
        st.success(f"Successfully added {person_name} to authorized users!")
        return True
    
    except Exception as e:
        st.error(f"Error capturing face: {str(e)}")
        return False
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()

# UI Layout
col1, col2 = st.columns([3, 2])

# Main video feed and controls
with col1:
    # Video feed placeholder
    video_placeholder = st.empty()
    
    # Control buttons in a row
    control_cols = st.columns(3)
    with control_cols[0]:
        system_button_text = "Stop System" if st.session_state.system_active else "Start System"
        if st.button(system_button_text, key="system_toggle"):
            if st.session_state.system_active:
                st.session_state.system_active = False
                stop_threads.set()
                st.session_state.system_status = "STANDBY"
            else:
                # Reset threads
                stop_threads.clear()
                st.session_state.system_active = True
                
                # Get available cameras
                cameras = get_available_cameras()
                camera_idx = cameras[0] if cameras else 0
                
                # Start video processing thread
                video_thread = threading.Thread(
                    target=process_video_feed, 
                    args=(camera_idx,),
                    daemon=True
                )
                video_thread.start()
                
                # Start sensor simulation thread if in simulation mode
                if st.session_state.simulation_mode:
                    import random
                    sensor_thread = threading.Thread(
                        target=simulate_sensors,
                        daemon=True
                    )
                    sensor_thread.start()
    
    with control_cols[1]:
        if st.button("Clear Alerts"):
            st.session_state.alert_message = ""
            st.session_state.alert_level = "info"
    
    with control_cols[2]:
        if st.button("Take Snapshot"):
            if st.session_state.current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"security_logs/snapshot_{timestamp}.jpg", 
                            cv2.cvtColor(st.session_state.current_frame, cv2.COLOR_RGB2BGR))
                log_security_event("Manual snapshot taken", "info", st.session_state.current_frame)
                st.success("Snapshot saved!")

# Status and information panel
with col2:
    # System status
    status_container = st.container()
    with status_container:
        st.subheader("System Status")
        
        # Status indicators
        st.metric("System Mode", "SIMULATION" if st.session_state.simulation_mode else "REAL SENSORS")
        
        status_cols = st.columns(2)
        with status_cols[0]:
            system_status_color = {
                "ACTIVE": "green",
                "STANDBY": "orange",
                "ERROR": "red",
                "INACTIVE": "gray"
            }.get(st.session_state.system_status, "gray")
            
            st.markdown(f"**System Status:** <span style='color:{system_status_color}'>{st.session_state.system_status}</span>", 
                      unsafe_allow_html=True)
            
            motion_status_color = "red" if st.session_state.motion_detected else "green"
            st.markdown(f"**Motion:** <span style='color:{motion_status_color}'>"
                      f"{'DETECTED' if st.session_state.motion_detected else 'CLEAR'}</span>", 
                      unsafe_allow_html=True)
        
        with status_cols[1]:
            door_status_color = "red" if st.session_state.door_status == "OPEN" else "green"
            st.markdown(f"**Door:** <span style='color:{door_status_color}'>{st.session_state.door_status}</span>", 
                      unsafe_allow_html=True)
            
            recognition_status = "ACTIVE" if st.session_state.face_recognition_active else "DISABLED"
            recognition_color = "green" if st.session_state.face_recognition_active else "gray"
            st.markdown(f"**Recognition:** <span style='color:{recognition_color}'>{recognition_status}</span>", 
                      unsafe_allow_html=True)
    
    # Alert box
    alert_container = st.container()
    with alert_container:
        if st.session_state.alert_message:
            if st.session_state.alert_level == "error":
                st.error(st.session_state.alert_message)
            elif st.session_state.alert_level == "warning":
                st.warning(st.session_state.alert_message)
            else:
                st.info(st.session_state.alert_message)
    
    # Recent events log
    st.subheader("Recent Events")
    log_container = st.container(height=300)
    with log_container:
        if st.session_state.security_logs:
            for log in reversed(st.session_state.security_logs[-10:]):
                event_time = log["timestamp"]
                event_msg = log["event"]
                event_level = log["level"]
                
                if event_level == "error":
                    st.error(f"{event_time}: {event_msg}")
                elif event_level == "warning":
                    st.warning(f"{event_time}: {event_msg}")
                else:
                    st.info(f"{event_time}: {event_msg}")
        else:
            st.write("No events recorded yet.")

# Sidebar for settings and management
with st.sidebar:
    st.title("Security Settings")
    
    # Face recognition settings
    st.subheader("Face Recognition")
    st.session_state.face_recognition_active = st.toggle("Enable Face Recognition", 
                                                       value=st.session_state.face_recognition_active)
    
    # Add new authorized person
    st.subheader("Add Authorized Person")
    new_person_name = st.text_input("Person Name")
    
    available_cameras = get_available_cameras()
    camera_options = {f"Camera {i}": i for i in available_cameras}
    
    if camera_options:
        selected_camera = st.selectbox("Select Camera", list(camera_options.keys()))
        camera_idx = camera_options[selected_camera]
        
        if st.button("Capture Face") and new_person_name:
            capture_authorized_face(camera_idx, new_person_name)
    else:
        st.error("No cameras detected")
    
    # Show authorized persons
    st.subheader("Authorized Persons")
    
    if st.session_state.authorized_faces:
        for name in st.session_state.authorized_faces.keys():
            st.write(f"✓ {name}")
    else:
        st.write("No authorized persons added yet.")
    
    # Advanced settings
    st.subheader("Advanced Settings")
    
    st.session_state.simulation_mode = st.toggle("Simulation Mode", value=st.session_state.simulation_mode,
                                              help="Use simulated sensors for testing")
    
    st.session_state.motion_sensitivity = st.slider("Motion Sensitivity", 0.1, 1.0, 
                                                  st.session_state.motion_sensitivity,
                                                  help="Higher values make motion detection more sensitive")
    
    if st.button("Export Security Logs"):
        # Create a DataFrame from the logs and convert to CSV
        import pandas as pd
        df = pd.DataFrame(st.session_state.security_logs)
        csv = df.to_csv(index=False)
        
        # Create a download button
        st.download_button(
            "Download CSV",
            csv,
            "security_logs.csv",
            "text/csv",
            key="download-csv"
        )

# Main update loop for the UI
if st.session_state.current_frame is not None:
    video_placeholder.image(st.session_state.current_frame, channels="RGB", use_column_width=True)
else:
    video_placeholder.info("Security camera inactive. Click 'Start System' to activate.")

# Initialize the system with a sample person if none exists
if not os.path.exists("authorized_faces/face_db.pkl") and not st.session_state.authorized_faces:
    # Load face database
    st.session_state.authorized_faces = load_face_database()
    
    # Set up simulation mode by default for first run
    st.session_state.simulation_mode = True