import cv2
import time
from picamera2 import Picamera2
from gpiozero import Servo

# =====================================================================
# 1. CLEAN HARDWARE CONFIGURATION (Pure Native Backend)
# =====================================================================
GPIO1 = 12
GPIO2 = 13
minPWM = 1/1000  # 1ms
maxPWM = 2/1000  # 2ms

# Declared directly without try/except so gpiozero automatically uses lgpio
Servo1 = Servo(GPIO1, min_pulse_width=minPWM, max_pulse_width=maxPWM)
Servo2 = Servo(GPIO2, min_pulse_width=minPWM, max_pulse_width=maxPWM)

#Servo1 = PWMOutputDevice(GPIO1, frequency=50)
#Servo2 = PWMOutputDevice(GPIO2, frequency=50)

STOP_DUTY = 0
# =====================================================================
# 2. DYNAMIC MOVEMENT FUNCTIONS
# =====================================================================
def track_move(error_x):
    """
    Dynamically adjusts servo speeds based on the tracking error.
    error_x ranges from -1.0 (far left) to 1.0 (far right).
    """
    deadzone = 0.25     # Ignore tiny movements to prevent jitter
    base_track_speed = 0.6 # Cap the maximum turning speed
    #base_track_modifier = 0.02
    
    if abs(error_x) > deadzone:
        # Calculate dynamic speed proportional to the distance from center
        speed = base_track_speed * abs(error_x)
        speed = max(0.15, min(speed, base_track_speed))
        #speed = 0.6
        #speed_offset = base_track_modifier * error_x
        
        if error_x > 0:
            # Face is to the RIGHT -> Turn Right
            print("RIGHT", speed)
            Servo1.value = speed
            Servo2.value = speed
        else:
            # Face is to the LEFT -> Turn Left
            print("LEFT", speed)
            Servo1.value = -speed
            Servo2.value = -speed
    else:
        stop_robot()

def stop_robot():
	if Servo1.value is not None or Servo1.value != 0: 
		Servo1.value = None
	if Servo2.value is not None or Servo2.value != 0: 
		Servo2.value = None
    
stop_robot()
# =====================================================================
# 3. VISION & TRACKING PIPELINE
# =====================================================================
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

picam = Picamera2()
FRAME_WIDTH, FRAME_HEIGHT = 1640, 1232  # Full sensor 4:3 FoV
config = picam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"})
picam.configure(config)
picam.start()

# Pre-calculate the exact center of your camera canvas
FRAME_CENTER_X = FRAME_WIDTH // 2
FRAME_CENTER_Y = FRAME_HEIGHT // 2

window_name = 'Noa Face Tracking'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 

print("\nTracking system active... Press 'q' to safely quit.")

try:
    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, -1) # Flips the image if the camera is upside down
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale 4x for ultra-fast processing matrix calculation
        scale_factor = 4
        small_gray = cv2.resize(gray, (0, 0), fx=(1/scale_factor), fy=(1/scale_factor))
        
        faces = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5)
        
        face_detected = False
        stop_robot()
        
        for (x, y, w, h) in faces:
            # Scale coordinates back up to draw bounding boxes accurately
            x, y, w, h = x * scale_factor, y * scale_factor, w * scale_factor, h * scale_factor
            
            # Calculate the horizontal center of the target face
            face_center_x = x + (w // 2)
            face_detected = True
            
            # Calculate the normalized error (-1.0 to 1.0)
            error_x = (face_center_x - FRAME_CENTER_X) / FRAME_CENTER_X
            
            # Send movement commands to continuous servos
            track_move(error_x)

            # Visual Feedback: Blue bounding box around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Red center point tracking dot
            cv2.circle(frame, (face_center_x, y + (h // 2)), 5, (0, 0, 255), -1)
            
            # Lock onto the first face found to avoid target confusion
            break 
            
        # If no faces are visible, stop the robot immediately so it doesn't spin endlessly
        if not face_detected:
            stop_robot()
            
        # Visual crosshair in the center of the frame (Green)
        cv2.line(frame, (FRAME_CENTER_X - 20, FRAME_CENTER_Y), (FRAME_CENTER_X + 20, FRAME_CENTER_Y), (0, 255, 0), 2)
        cv2.line(frame, (FRAME_CENTER_X, FRAME_CENTER_Y - 20), (FRAME_CENTER_X, FRAME_CENTER_Y + 20), (0, 255, 0), 2)
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safe cleanup block to guarantee the robot cuts power when you stop the script
    print("\nShutting down hardware interfaces...")
    stop_robot()
    picam.stop()
    cv2.destroyAllWindows()
