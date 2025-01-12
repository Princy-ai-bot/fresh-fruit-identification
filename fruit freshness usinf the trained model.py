import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the saved model
model_path = r'C:\Users\yashs\Desktop\flipkart\fruit_freshness_model_mobilenet.keras'  # Update this with your saved model path
model = load_model(model_path)

# Function to preprocess image for live prediction
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match input shape of the model
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale pixel values
    return image

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    print("Camera opened successfully. Show a fruit to the camera for prediction.")

# Class labels (Ensure these match the labels used during training)
classes = ['freshapples', 'freshoranges', 'freshbanana', 'rottenapples', 'rottenoranges', 'rottenbanana']

# Loop for continuous camera input and prediction
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame for prediction
    processed_image = preprocess_image(frame)

    # Predict using the model
    prediction = model.predict(processed_image)
    pred_class = np.argmax(prediction[0])  # Get the predicted class index

    # Put the predicted class label on the frame
    label = classes[pred_class]
    cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with the label
    cv2.imshow('Fruit Freshness Detector', frame)

    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
