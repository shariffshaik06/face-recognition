# face-recognition
The "Real-time Face and Eye Detection using OpenCV" project is a computer vision application that leverages the power of OpenCV.This project showcases fundamental concepts in object detection and serves as a foundation for more advanced computer vision applications.

#firstly import the cv2 library
import cv2

# Load the cascades
face_cascade = cv2.CascadeClassifier('C:\\Users\\shariff shaik\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\shariff shaik\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml')

# To capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around each face and detect eyes in the face region
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (225, 0 ,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # Draw rectangles around each eye
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()

