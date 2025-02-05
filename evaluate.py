import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from deepface import DeepFace
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Paths
db_path = "dataset"  # Directory containing known faces
test_path = "test_data"  # Directory containing test images
knn_model_path = "data/knn_model.pkl"  # KNN model path
names_path = "data/names.pkl"  # Names associated with KNN labels

# Initialize variables
true_labels = []
predicted_labels_deepface = []
predicted_labels_knn = []

# Load KNN Model and Names
with open(knn_model_path, "rb") as f:
    knn_model = pickle.load(f)

with open(names_path, "rb") as f:
    knn_names = pickle.load(f)

def predict_face_deepface(img_path):
    """
    Function to predict face using DeepFace.
    Returns the recognized name or 'Unknown'.
    """
    try:
        result = DeepFace.find(img_path=img_path, db_path=db_path, model_name="Facenet", enforce_detection=False)
        if len(result) > 0:
            # Extract the recognized name from the identity path
            recognized_name = result[0]["identity"][0].split(os.sep)[-2]
            return recognized_name
        else:
            return "Unknown"
    except Exception as e:
        print(f"Prediction error (DeepFace): {e}")
        return "Unknown"

def predict_face_knn(img_path):
    """
    Function to predict face using KNN.
    Returns the recognized name or 'Unknown'.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50)).flatten().reshape(1, -1)
    try:
        predicted_label = knn_model.predict(img)[0]
        return predicted_label
    except Exception as e:
        print(f"Prediction error (KNN): {e}")
        return "Unknown"

# Load Test Data
print("Starting Evaluation...")
for person_name in os.listdir(test_path):  # Iterate through each person's folder
    person_folder = os.path.join(test_path, person_name)

    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):  # Iterate through each image
            img_path = os.path.join(person_folder, img_name)

            # Predict using DeepFace
            predicted_name_deepface = predict_face_deepface(img_path)
            predicted_labels_deepface.append(predicted_name_deepface)

            # Predict using KNN
            predicted_name_knn = predict_face_knn(img_path)
            predicted_labels_knn.append(predicted_name_knn)

            # Store the true label
            true_labels.append(person_name)

            print(f"Image: {img_name}, True: {person_name}, DeepFace: {predicted_name_deepface}, KNN: {predicted_name_knn}")

# Generate Confusion Matrix and Performance Metrics
print("\nEvaluation Results:")

# Unique labels for both models
labels = list(set(true_labels)) + ["Unknown"]

# DeepFace Metrics
print("\nConfusion Matrix (DeepFace):")
cm_deepface = confusion_matrix(true_labels, predicted_labels_deepface, labels=labels)
print(cm_deepface)

print("\nClassification Report (DeepFace):")
print(classification_report(true_labels, predicted_labels_deepface, labels=labels))

accuracy_deepface = accuracy_score(true_labels, predicted_labels_deepface)
print(f"Overall Accuracy (DeepFace): {accuracy_deepface:.2f}")

# KNN Metrics
print("\nConfusion Matrix (KNN):")
cm_knn = confusion_matrix(true_labels, predicted_labels_knn, labels=labels)
print(cm_knn)

print("\nClassification Report (KNN):")
print(classification_report(true_labels, predicted_labels_knn, labels=labels))

accuracy_knn = accuracy_score(true_labels, predicted_labels_knn)
print(f"Overall Accuracy (KNN): {accuracy_knn:.2f}")

# Visualization
plt.figure(figsize=(12, 6))

# DeepFace Confusion Matrix Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(cm_deepface, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (DeepFace)")
plt.xlabel("Predicted")
plt.ylabel("True")

# KNN Confusion Matrix Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (KNN)")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()

# Accuracy Comparison Bar Graph
models = ['DeepFace', 'KNN']
accuracies = [accuracy_deepface, accuracy_knn]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Calculate and Plot Cumulative Accuracy for DeepFace Model
cumulative_accuracies = []
correct_predictions = 0

for i in range(len(true_labels)):
    # Update correct predictions count
    if true_labels[i] == predicted_labels_deepface[i]:
        correct_predictions += 1
    # Calculate cumulative accuracy
    cumulative_accuracies.append(correct_predictions / (i + 1))

# Plot the Line Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(true_labels) + 1), cumulative_accuracies, label='DeepFace Cumulative Accuracy', color='blue', linewidth=2)
plt.axhline(y=accuracy_deepface, color='red', linestyle='--', label=f'Final Accuracy: {accuracy_deepface:.2f}')
plt.title("DeepFace Model Performance Over Test Set")
plt.xlabel("Number of Test Samples")
plt.ylabel("Cumulative Accuracy")
plt.legend()
plt.grid(True)
plt.show()

