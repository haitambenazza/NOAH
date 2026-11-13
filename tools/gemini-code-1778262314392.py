import cv2
import numpy as np
from picamera2 import Picamera2

# 1. Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 2. Initialize the modern Picamera2 hardware interface
picam = Picamera2()

# OPTIMIZATION FOR MAXIMUM FOV:
# Using 1640x1232 (4:3) utilizes the full physical sensor instead of cropping to 16:9.
config = picam.create_preview_configuration(main={"size": (1640, 1232), "format": "BGR888"})
picam.configure(config)

# Start the internal camera thread stream
picam.start()

# 3. Setup display window
window_name = 'Face Detection - Full Sensor FoV'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

print("Starting camera feed with Max FoV... Press 'q' to quit.")

try:
    while True:
        # Capture the full-sensor 1640x1232 resolution frame
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Flip frame if the camera module is mounted upside down
        frame = cv2.flip(frame, -1)
        
        # Convert to grayscale for calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # PERFORMANCE OPTIMIZATION: 
        # Since 1640x1232 is larger, downscale by 4x (to 410x308) so the CPU scans instantly
        scale_factor = 4
        small_gray = cv2.resize(gray, (0, 0), fx=(1/scale_factor), fy=(1/scale_factor))
        
        # Run face detection on the highly optimized small matrix
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=4)
        
        # Scale coordinates back up by 4x to draw accurately on the full-size frame
        for (x, y, w, h) in faces:
            x, y, w, h = x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Render the frame to our specific expanded named window
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nClosing camera interface gracefully...")
    picam.stop()
    cv2.destroyAllWindows()
