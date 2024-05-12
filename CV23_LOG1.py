import cv2
import subprocess
import time

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(1)

face_detected = False

while True:
    # Capture a frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Set face_detected flag if a face is detected
        if not face_detected:
            face_detected = True

    # Display the live feed with detected faces
    cv2.imshow('Face Recognition', frame)

    # Break the loop if 'q' is pressed or a face is detected
    if cv2.waitKey(1) & 0xFF == ord('q') or face_detected:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()

# Open the index2.html file in a web browser after the OpenCV window is closed
if face_detected:
    subprocess.call(['open', 'index2.html'])
    time.sleep(3)  # Wait for 3 seconds