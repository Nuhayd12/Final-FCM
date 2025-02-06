import cv2
from deepface import DeepFace
from picamera2 import Picamera2

# Initialize Raspberry Pi Camera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
camera.start()

# Initialize face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def recognize_face(image):
    """
    Recognize face using DeepFace.
    """
    try:
        result = DeepFace.find(img_path=image, db_path="dataset", model_name="Facenet")
        if len(result) > 0:
            return result[0]['identity'].split('/')[-2]  # Extract person's name from file path
        else:
            return "Unknown"
    except Exception as e:
        print(f"Recognition error: {e}")
        return "Unknown"

print("Starting face recognition...")
while True:
    img = camera.capture_array()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        name = recognize_face(face_img)

        # Display the recognized name
        cv2.putText(img, f"Hi, {name}!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
