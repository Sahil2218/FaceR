import cv2
from mtcnn.mtcnn import MTCNN
import os

# Initialize the MTCNN detector
detector = MTCNN()

# Create a directory to save the captured faces
output_directory = 'captured_faces_mtcnn'
os.makedirs(output_directory, exist_ok=True)

# Open the video stream (replace '0' with your video file path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Loop through detected faces and draw rectangles
    for result in faces:
        x, y, w, h = result['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Capture and save the detected face
        face_image = frame[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(output_directory, f'face_{len(os.listdir(output_directory)) + 1}.jpg'), face_image)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()