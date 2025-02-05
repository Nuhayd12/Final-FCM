import cv2
import serial
from deepface import DeepFace

# Initialize the camera
video = cv2.VideoCapture(0)

# Load the face detection model
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Establish serial communication with Arduino
# Replace '/dev/ttyUSB0' with the correct port on your Raspberry Pi
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

def send_command_to_arduino(command):
    """Send a command to the Arduino."""
    arduino.write(f"{command}\n".encode())
    print(f"Sent to Arduino: {command}")

# Main logic
print("Starting face recognition bot control system...")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    face_detected = False
    recognized_name = None

    for (x, y, w, h) in faces:
        face_detected = True
        crop_img = frame[y:y+h, x:x+w]
        
        try:
            # Perform face recognition
            result = DeepFace.find(crop_img, db_path="faculty_images", enforce_detection=False, silent=True)
            if not result.empty:
                recognized_name = result.iloc[0]['identity'].split('/')[-2]  # Extract the name
        except Exception as e:
            print(f"Error during recognition: {e}")
            recognized_name = None

        # Display the face box and name on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label = f"Hi, {recognized_name}!" if recognized_name else "Unknown"
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Control the bot based on face detection
    if face_detected:
        if recognized_name:
            send_command_to_arduino("STOP")  # Stop the bot if a face is recognized
            print(f"Recognized: {recognized_name}")
        else:
            send_command_to_arduino("STOP")  # Also stop for unrecognized faces
            print("Face detected but not recognized.")
    else:
        send_command_to_arduino("START")  # Start the bot if no face is detected

    # Show the camera feed
    cv2.imshow("Face Recognition Bot Control", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
arduino.close()
print("System shut down.")
