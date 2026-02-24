import cv2
import face_recognition
import pickle
import numpy as np
import pyttsx3
import torch
from ultralytics import YOLO
from queue import Queue
from threading import Thread
from inference import InferencePipeline

# Constants
ENCODINGS_FILE = "face_encodings.pkl"
DISAPPEARANCE_THRESHOLD = 30  # Frames to consider an object "gone"

# Load Face Encodings
def load_face_encodings():
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
tts_queue = Queue()

# TTS Functions
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text == "STOP":
            break
        speak_text(text)

tts_thread = Thread(target=tts_worker, daemon=True)
tts_thread.start()

# YOLO Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO('yolov8s.pt').to(device)

# InferencePipeline Callback
def inference_callback(result, video_frame):
    if result.get("output_image"):
        cv2.imshow("Inference Image", result["output_image"].numpy_image)
        cv2.waitKey(1)

    predictions = result.get("predictions", [])
    for prediction in predictions:
        label = prediction.get("class", "Unknown")
        confidence = prediction.get("confidence", 0)
        message = f"Detected: {label} ({confidence:.2f})"
        print(message)
        tts_queue.put(message)

# Initialize InferencePipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key="9p0CEWcQsJC7euMxImuU",
    workspace_name="shadil-ibrahim-2klvc",
    workflow_id="detect-count-and-visualize",
    video_reference=0,
    max_fps=30,
    on_prediction=inference_callback
)

# Combined Recognition Function
def recognize_faces_and_objects():
    data = load_face_encodings()
    video_capture = cv2.VideoCapture(0)

    detected_faces = set()
    detected_objects = {}

    if not video_capture.isOpened():
        print("Error: Cannot access webcam.")
        return

    pipeline.start()  # Start InferencePipeline

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        current_faces = set()
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]

            current_faces.add(name)
            if name not in detected_faces:
                tts_queue.put(f"{name} detected")
                detected_faces.add(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        detected_faces.intersection_update(current_faces)

        # YOLO Object Detection
        results = model(frame[..., ::-1])
        current_objects = set()

        for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                label_index = int(box.cls[0])
                label_name = model.names.get(label_index, "Unknown")
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if confidence >= 0.7:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label_name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    current_objects.add(label_name)

                    if label_name not in detected_objects:
                        tts_queue.put(f"Detected: {label_name}")
                        detected_objects[label_name] = 0
                    else:
                        detected_objects[label_name] = 0

        for obj in list(detected_objects.keys()):
            if obj not in current_objects:
                detected_objects[obj] += 1
                if detected_objects[obj] > DISAPPEARANCE_THRESHOLD:
                    del detected_objects[obj]

        cv2.imshow("Face & Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    pipeline.join()
    cv2.destroyAllWindows()
    tts_queue.put("STOP")

# Main Function
def main():
    recognize_faces_and_objects()
    tts_thread.join()

if __name__ == "__main__":
    main()
