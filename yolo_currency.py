import cv2
import pyttsx3
from queue import Queue
from threading import Thread

# Import the InferencePipeline object
from inference import InferencePipeline

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Queue for TTS messages
tts_queue = Queue()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text == "STOP":
            break
        speak_text(text)

# Start TTS thread
tts_thread = Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Callback function to handle predictions
def my_sink(result, video_frame):
    if result.get("output_image"):  # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)

    predictions = result.get("predictions", [])
    for prediction in predictions:
        label = prediction.get("class", "Unknown")
        confidence = prediction.get("confidence", 0)
        detection_info = f"Detected: {label} ({confidence:.2f})"
        print(detection_info)
        tts_queue.put(detection_info)
        tts_queue.put(f"Announcement: {label} detected with {confidence * 100:.1f} percent confidence")  # Audio feedback

# Initialize the InferencePipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="9p0CEWcQsJC7euMxImuU",
    workspace_name="shadil-ibrahim-2klvc",
    workflow_id="detect-count-and-visualize",
    video_reference=0,  # Path to video, device ID (int, usually 0 for built-in webcams), or RTSP stream URL
    max_fps=30,
    on_prediction=my_sink
)

# Start the pipeline
pipeline.start()

# Wait for the pipeline thread to finish
pipeline.join()

tts_queue.put("STOP")
tts_thread.join()
