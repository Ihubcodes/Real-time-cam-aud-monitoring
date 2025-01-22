import cv2
import numpy as np
import time
from PIL import Image
from datetime import datetime
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
import sounddevice as sd

# Set device for torch
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Initialize MTCNN for face detection
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).eval()

# Initialize InceptionResnetV1 model for embedding extraction
model = InceptionResnetV1(pretrained="vggface2", classify=False).to(DEVICE).eval()

# Define folder paths for saving alert images and video recordings
alert_folder_path = r"D:\phi-2-chatbot\uploads\images"
video_folder_path = r"D:\phi-2-chatbot\uploads\videos"
alert_log_file = r"D:\phi-2-chatbot\alerts_log.txt"

# Ensure folder paths exist
os.makedirs(alert_folder_path, exist_ok=True)
os.makedirs(video_folder_path, exist_ok=True)

# Function to detect conditions (multiple persons, no person, focus, screen cover) in the frame
def detect_conditions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    alerts = []

    if len(faces) == 0:
        alerts.append("No person found")
    elif len(faces) > 1:
        alerts.append("Multiple persons in frame")
    else:
        (x, y, w, h) = faces[0]
        face_region = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        if len(eyes) == 0:
            alerts.append("Focus on the screen")
        else:
            for (ex, ey, ew, eh) in eyes:
                eye_center_x = x + ex + ew // 2
                eye_center_y = y + ey + eh // 2
                if not (x + w * 0.25 < eye_center_x < x + w * 0.75):
                    alerts.append("Focus on the screen")
                    break

        avg_brightness = np.mean(face_region)
        if avg_brightness < 30 or avg_brightness > 220:
            alerts.append("Screen is covered")

    # If no alerts, add "Valid frame" message
    if not alerts:
        alerts.append("Valid frame")

    return faces, alerts

# Function to capture reference image from webcam
def capture_reference_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam.")
        return None
    else:
        # Call the condition detection function here
        faces, alerts = detect_conditions(frame)

        # Print or log the alerts if any
        for alert in alerts:
            print(alert)

        # Convert the frame to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reference_image = Image.fromarray(frame_rgb)

        cap.release()
        cv2.destroyAllWindows()
        return reference_image

# Preprocess the face for embedding extraction
def preprocess(face):
    face = face.unsqueeze(0)  # Add batch dimension
    face = F.interpolate(face, size=(160, 160), mode='bilinear', align_corners=False)
    face = face.to(DEVICE).to(torch.float32) / 255.0
    return face

# Get embedding for a face image
def get_embedding(input_image):
    face = mtcnn(input_image)
    if face is None:
        return None
    face = preprocess(face)
    with torch.no_grad():
        embedding = model(face).cpu().numpy().flatten()
    return embedding

# Compare embeddings to verify face match
def compare_faces(embedding1, embedding2, threshold=0.6):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance, distance < threshold

# Save alert image to the specified folder with timestamp
def save_alert_image(frame, alert_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(alert_folder_path, f"{alert_type}_{timestamp}.jpg")
    cv2.imwrite(image_path, frame)
    return image_path

# Save alert message to a log file
def log_alert(message):
    with open(alert_log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Audio detection function using sounddevice
def detect_sound(threshold=0.01, duration=0.5, fs=44100):
    """Detects sound based on threshold."""
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    energy = np.sum(np.abs(audio_data) ** 2)
    return energy > threshold

# Function to run the face verification and monitoring process
def run_monitoring(reference_image, reference_embedding):
    cap = cv2.VideoCapture(0)
    alert_log = []
    start_verification = False

    # Define video writer for recording
    video_filename = os.path.join(video_folder_path, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 2.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # Set the initial time for monitoring
    last_capture_time = time.time()
    face_alert_count = 0
    multiple_faces_alert_count = 0
    sound_alert_count = 0

    while cap.isOpened():
        current_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to access webcam.")
            break

        # Capture frame every 5 seconds
        if current_time - last_capture_time >= 5:
            last_capture_time = current_time  # Update last capture time
            out.write(frame)

            # Convert frame for embedding extraction and verification
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_image = Image.fromarray(frame_rgb)

            # Detect faces in the current frame
            faces, _ = mtcnn.detect(current_image)
            if faces is None or len(faces) == 0:
                face_alert_count += 1
                alert_message = "No face detected in the current frame."
                print(alert_message)
                alert_log.append((datetime.now(), alert_message))
                save_alert_image(frame, "no_face")
                log_alert(alert_message)
            else:
                multiple_faces_detected = len(faces) > 1
                current_embedding = get_embedding(current_image)

                if current_embedding is None:
                    face_alert_count += 1
                    alert_message = "No face detected in the current frame."
                    print(alert_message)
                    alert_log.append((datetime.now(), alert_message))
                    save_alert_image(frame, "no_face")
                    log_alert(alert_message)
                else:
                    # Compare with reference embedding
                    distance, is_match = compare_faces(reference_embedding, current_embedding)

                    if is_match:
                        if not start_verification:
                            print("Face Verified Successfully.")
                        start_verification = True
                        print("Monitoring candidate's presence...")
                    else:
                        face_alert_count += 1
                        alert_message = "Unrecognized face detected in the frame."
                        print(alert_message)
                        alert_log.append((datetime.now(), alert_message))
                        save_alert_image(frame, "face_mismatch")
                        log_alert(alert_message)

                    # If multiple faces are detected, log an alert
                    if multiple_faces_detected:
                        multiple_faces_alert_count += 1
                        alert_message = "Multiple faces detected!"
                        print(alert_message)
                        alert_log.append((datetime.now(), alert_message))
                        save_alert_image(frame, "multiple_faces")
                        log_alert(alert_message)

            # Detect if there is loud noise in the environment
            if detect_sound():
                sound_alert_count += 1
                alert_message = "Loud noise detected in the environment."
                print(alert_message)
                alert_log.append((datetime.now(), alert_message))
                save_alert_image(frame, "noise_detected")
                log_alert(alert_message)

        # End monitoring if alert count exceeds 3 for any alert type
        if face_alert_count >= 3 or multiple_faces_alert_count >= 3 or sound_alert_count >= 3:
            print("Monitoring ended due to excessive alerts.")
            break

        # Draw bounding boxes and alerts on the frame
        # ... (Add this part below)

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Capture reference image from webcam
    reference_image = capture_reference_image()

    # If reference image is captured successfully
    if reference_image is not None:
        print("Reference image captured successfully.")
        # Extract embedding from the captured reference image
        reference_embedding = get_embedding(reference_image)
        if reference_embedding is None:
            print("No face detected in the reference image.")
        else:
            print("Reference embedding extracted successfully.")
            # Run face verification and monitoring
            run_monitoring(reference_image, reference_embedding)