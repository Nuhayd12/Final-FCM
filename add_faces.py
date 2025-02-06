import os
import cv2
import numpy as np
from picamera2 import Picamera2
from deepface import DeepFace

# Initialize Raspberry Pi Camera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
camera.start()

# Initialize face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create the dataset directory if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Input for person name
name = input("Enter Your Name: ")
person_dir = f"dataset/{name}"
if not os.path.exists(person_dir):
    os.makedirs(person_dir)

i = 0  # Counter for saved images

print("Starting data collection...")
while True:
    img = camera.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (224, 224))  # Resize for DeepFace

        # Save face images every 10 frames
        if i % 10 == 0 and i // 10 < 100:
            face_path = os.path.join(person_dir, f"face_{i//10}.jpg")
            cv2.imwrite(face_path, resized_face)
            print(f"Saved {face_path}")

        i += 1

        # Display progress
        cv2.putText(img, f"Collecting: {i//10}/100", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Data Collection", img)

    if cv2.waitKey(1) & 0xFF == ord('q') or i // 10 >= 100:
        break

print("Data collection completed!")
cv2.destroyAllWindows()
