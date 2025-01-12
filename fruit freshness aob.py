import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import Mobile
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrixqqqq
NetV2
from tensorflow.keras.models import Sequential
# Define paths for the dataset
train_dir = "C:\\Users\\yashs\\Desktop\\dataset\\train"
test_dir = "C:\\Users\\yashs\\Desktop\\dataset\\test"

# Image Data Generators for loading the data and applying augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,       # Normalization
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training data from the train directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # MobileNetV2 requires 224x224 input size
    batch_size=32,           # You can adjust this depending on memory
    class_mode='categorical', # Categorical classification for fresh and rotten classes
    shuffle=True              # Shuffle the data
)

# Load testing data from the test directory
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the MobileNetV2 model with pretrained weights from ImageNet and exclude the top layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),   # Global pooling layer to reduce dimensions
    Dense(128, activation='relu'),
    Dropout(0.5),               # Dropout for regularization
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer for categorical classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to reduce learning rate if the model plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model with callback for learning rate reduction
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size,
    epochs=10,  # Set this higher to observe stability
    callbacks=[reduce_lr],
    verbose=1   # Print detailed logs
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.4f}')

# Confusion Matrix and Classification Report
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = list(train_generator.class_indices.keys())  # Automatically get class names from the generator
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# Save the trained model in Keras format
model.save('fruit_freshness_model_mobilenet.keras')  # Changed to .keras extension

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

# Class labels
classes = list(train_generator.class_indices.keys())

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
