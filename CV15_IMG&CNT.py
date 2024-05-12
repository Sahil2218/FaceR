import cv2
import os

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to save the captured faces
output_directory = 'captured_faces'
os.makedirs(output_directory, exist_ok=True)

# Open the video stream (replace '0' with your video file path)
cap = cv2.VideoCapture(0)

# Initialize face counter
face_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through detected faces and draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Capture and save the detected face
        face_image = frame[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(output_directory, f'face_{face_counter + 1}.jpg'), face_image)

        # Increment face counter
        face_counter += 1

    # Display the frame with the live counter
    cv2.putText(frame, f'Faces Detected: {face_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
