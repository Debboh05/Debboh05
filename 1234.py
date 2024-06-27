# Import necessary libraries
import cv2
import numpy as np

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained emotion recognition model using CNN
# Model training and loading code here

# Function to detect faces and classify emotions
def detect_emotions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        # Preprocess face_roi for emotion recognition
        # Emotion classification using CNN model
        predicted_emotion = model.predict(face_roi)
        
        # Draw bounding box around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    return image

# Load input image or frame
input_image = cv2.imread('input_image.jpg')

# Detect emotions in the input image
output_image = detect_emotions(input_image)

# Display the output image with emotion labels
cv2.imshow('Facial Expression Recognition', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
