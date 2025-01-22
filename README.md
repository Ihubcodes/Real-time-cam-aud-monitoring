# Face Monitoring and Verification System

This project implements a real-time face monitoring and verification system using webcam input. The system ensures the user's presence, detects environmental conditions, and logs relevant alerts when anomalies occur.

## Features
1. **Face Verification**: Compares the reference face with the faces detected during monitoring.
2. **Condition Detection**: Identifies multiple people, no person, lack of focus, and screen obstruction.
3. **Sound Detection**: Monitors ambient noise levels and triggers an alert for loud noises.
4. **Alert Management**:
   - Captures and saves frames for triggered alerts.
   - Logs alerts to a text file with timestamps.
5. **Recording**: Saves the video feed during the monitoring session.

---

## Installation and Setup

### Requirements
1. Python 3.7+
2. Required Python libraries:
   - `opencv-python`
   - `numpy`
   - `torch`
   - `facenet-pytorch`
   - `pillow`
   - `sounddevice`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Additional Configuration
Ensure the directories specified in the code for saving alert images and video recordings exist. Default paths can be found in the script:
- Images: `D:\phi-2-chatbot\uploads\images`
- Videos: `D:\phi-2-chatbot\uploads\videos`

You can modify these paths as needed.

---

## How to Use

### Reference Image Capture
1. Run the script:
   ```bash
   python script_name.py
   ```
2. A reference image will be captured automatically via the webcam.
3. Ensure your face is clearly visible during reference capture.

### Monitoring
1. After the reference image is captured, the system extracts the face embedding.
2. Real-time monitoring begins, during which the following are tracked:
   - Face verification
   - Number of people in the frame
   - Sound levels
   - Focus and environmental obstructions

### Alerts
- Alerts are logged to a file (`alerts_log.txt`), and snapshots of the frame are saved in the `images` folder for visual evidence.
- If excessive alerts are detected for a condition (3 consecutive occurrences), monitoring ends automatically.

### Exit
To manually end monitoring, press the `q` key in the camera feed window.

---

## Alert Logging
Alerts are saved with detailed descriptions and timestamps in:
```text
D:\phi-2-chatbot\alerts_log.txt
```
The alerts may include:
- **No face detected**: No person found in the frame.
- **Unrecognized face**: Face does not match the reference image.
- **Multiple faces detected**: More than one person in the frame.
- **Focus issue**: User not focusing on the screen or eyes not detected properly.
- **Loud noise detected**: Unusual noise levels in the environment.

---

## File Descriptions
1. **Python Script**: Contains the main logic for face verification and monitoring.
2. **Alert Log File**: Tracks alert messages with timestamps.
3. **Captured Images**: Saves frames associated with specific alerts.
4. **Recorded Videos**: Contains the video feed for the session.

---

## Limitations
1. Limited to single-camera setups.
2. Environmental factors (e.g., lighting) may impact face detection accuracy.
3. Alerts rely on fixed thresholds, which may require adjustments for specific use cases.

---

