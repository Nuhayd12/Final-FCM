import os
import cv2
import pickle
import numpy as np
from picamera2 import Picamera2
from sklearn.neighbors import KNeighborsClassifier
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
face_embeddings = []  # To store embeddings for KNN and DeepFace
face_labels = []  # To store corresponding labels

def preprocess_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Apply histogram equalization
    return equalized

def extract_embedding(image_path):
    """
    Extract facial embedding using DeepFace.
    """
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="Facenet")
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return None

# Capture frames from the camera
print("Starting data collection...")
while True:
    img = camera.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        
        # Preprocess the face (grayscale and histogram equalization)
        processed_face = preprocess_face(face_img)
        resized_face = cv2.resize(processed_face, (50, 50))

        # Save face images every 10 frames
        if i % 10 == 0 and i // 10 < 100:
            face_path = os.path.join(person_dir, f"face_{i//10}.jpg")
            cv2.imwrite(face_path, resized_face)
            print(f"Saved {face_path}")

            # Extract embedding and store it
            embedding = extract_embedding(face_path)
            if embedding is not None:
                face_embeddings.append(embedding)
                face_labels.append(name)
        
        i += 1

        # Display the progress
        cv2.putText(img, f"Collecting: {i//10}/100", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Data Collection", img)

    if cv2.waitKey(1) & 0xFF == ord('q') or i // 10 >= 100:
        break

print("Data collection completed!")
cv2.destroyAllWindows()

# Train KNN Model
print("Training KNN model...")
face_embeddings = np.array(face_embeddings)
face_labels = np.array(face_labels)

knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_model.fit(face_embeddings, face_labels)

# Save KNN model and labels
if not os.path.exists("data"):
    os.makedirs("data")

with open("data/knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

with open("data/names.pkl", "wb") as f:
    pickle.dump(list(set(face_labels)), f)

print("KNN model trained and saved!")
