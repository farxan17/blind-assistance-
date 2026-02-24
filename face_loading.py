import os
import cv2
import face_recognition
import pickle

# Path to dataset
DATASET_PATH = "dataset"
ENCODINGS_FILE = "face_encodings.pkl"

# Initialize lists
known_encodings = []
known_names = []

# Loop through each person’s folder
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    # Process each image
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect and encode face
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        for encoding in face_encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("Face encodings saved!")
