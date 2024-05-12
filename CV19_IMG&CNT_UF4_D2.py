import cv2
import dlib
import os
import face_recognition

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Create a directory to save the captured faces
output_directory = 'captured_faces'
os.makedirs(output_directory, exist_ok=True)

# Initialize face counter and list to store face encodings
face_counter = 0
known_face_encodings = []
known_face_locations = []

# Variables for frame skipping
frame_skip = 10  # Process every 5th frame
current_frame = 0

# Open the video stream (replace '0' with your video file path)
cap = cv2.VideoCapture(0)

# Lower the frame resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1

    # Skip frames
    if current_frame % frame_skip != 0:
        continue

    # Find face locations in the frame using face_recognition
    face_locations = face_recognition.face_locations(frame)

    # Loop through detected face locations and draw rectangles
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Encode the face using face_recognition library
        face_encoding = face_recognition.face_encodings(frame, [face_location], num_jitters=1)

        # Check if the face is already captured
        if face_encoding:
            # Compare with known face encodings
            match = face_recognition.compare_faces(known_face_encodings, face_encoding[0])

            if not any(match):
                # Capture and save the detected face
                cv2.imwrite(os.path.join(output_directory, f'face_{face_counter + 1}.jpg'), frame[top:bottom, left:right])

                # Add the face encoding and location to the lists
                known_face_encodings.append(face_encoding[0])
                known_face_locations.append(face_location)

                # Draw rectangle on the frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Increment face counter
                face_counter += 1

    # Display the frame with the live counter
    cv2.putText(frame, f'Unique Faces Detected: {face_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
