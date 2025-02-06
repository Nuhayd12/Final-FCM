from flask import Flask, Response
import cv2
from picamera2 import Picamera2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Initialize Raspberry Pi Camera (Picamera2)
camera = Picamera2()
camera.configure(camera.create_video_configuration(main={"size": (640, 480)}))
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

def gen_frames():
    while True:
        try:
            frame = camera.capture_array()

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                name = recognize_face(face_img)

                cv2.putText(frame, f"Hi, {name}!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
