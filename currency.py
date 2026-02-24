import cv2
import numpy as np
import tensorflow as tf
import pyttsx3  # For text-to-speech

# Load the trained Keras model
model = tf.keras.models.load_model('keras_model.h5')

# Labels for Indian currency notes (update as per your model)
CURRENCY_LABELS = ['10 INR', '20 INR', '50 INR', '100 INR', '200 INR', '500 INR', '2000 INR']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# To store the last detected currency
last_detected = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for the model
    img = cv2.resize(frame, (224, 224))  # Adjust size based on your model
    img = img.astype(np.float32) / 255.0  # Normalization if required
    img = np.expand_dims(img, axis=0)

    # Prediction
    predictions = model.predict(img)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    currency = CURRENCY_LABELS[predicted_class]

    # Display detection only if it's different from the last detected
    if currency != last_detected and confidence > 0.9:  # Adjust confidence threshold as needed
        last_detected = currency
        print(f"Detected Currency: {currency} with {confidence * 100:.2f}% confidence")

        # Voice feedback
        engine.say(f"Detected {currency}")
        engine.runAndWait()

        # Save the frame
        cv2.imwrite(f"detected_{currency}.png", frame)

    # Display the frame with detected currency label
    cv2.putText(frame, f"Currency: {last_detected if last_detected else 'Detecting...'}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Indian Currency Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
