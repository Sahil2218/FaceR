import cv2
import dlib
import os

# Load the pre-trained face detector from dlib
detector = dlib.get_frontal_face_detector()

# Create a directory to save the captured faces
output_directory = 'captured_faces'
os.makedirs(output_directory, exist_ok=True)

# Open the video stream (replace '0' with your video file path)
cap = cv2.VideoCapture(0)

# Initialize face counter and dictionary to store unique face identifiers
face_counter = 0
unique_faces = {}

# Variables for frame skipping
frame_skip = 2  # Process every 2nd frame
current_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_frame += 1

    # Skip frames
    if current_frame % frame_skip != 0:
        continue

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using dlib
    faces = detector(gray)

    # Loop through detected faces and draw rectangles
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Check if the face is already captured
        face_id = f'{x}-{y}-{w}-{h}'
        if face_id not in unique_faces:
            cv2.rectangle(frame, (x * 2, y * 2), ((x + w) * 2, (y + h) * 2), (0, 255, 0), 2)

            # Capture and save the detected face
            face_image = frame[y * 2:(y + h) * 2, x * 2:(x + w) * 2]
            cv2.imwrite(os.path.join(output_directory, f'face_{face_counter + 1}.jpg'), face_image)

            # Add the face to the dictionary of unique faces
            unique_faces[face_id] = face_counter

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
