import cv2
import os
import face_recognition
import threading
import webbrowser

# Load the pre-trained face detector from face_recognition
known_face_encodings = []
known_face_names = []

# Create a directory to save the captured faces
output_directory = 'captured_faces'
os.makedirs(output_directory, exist_ok=True)

# Variables for frame skipping
frame_skip = 2  # Process every 2nd frame
current_frame = 0

# Lock for thread-safe access to shared data
lock = threading.Lock()

# Function for face recognition
def recognize_faces(frame):
    global known_face_encodings
    global known_face_names

    # Find face locations and encodings in the frame using face_recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding in face_encodings:
        # Check if the face is already captured
        match = face_recognition.compare_faces(known_face_encodings, face_encoding)

        if not any(match):
            # Capture and save the detected face
            face_name = input("Enter your name: ")
            known_face_encodings.append(face_encoding)
            known_face_names.append(face_name)

            # Save the face image
            cv2.imwrite(os.path.join(output_directory, f'{face_name}.jpg'), frame)

            # Open the login page in a web browser
            webbrowser.open('index2.html')

            # Close the OpenCV window
            cv2.destroyAllWindows()
            return

# Open the video stream (replace '0' with your video file path)
cap = cv2.VideoCapture(1)

# Lower the frame resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Function to read frames and start face recognition in a separate thread
def process_frames():
    global current_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        # Skip frames
        if current_frame % frame_skip != 0:
            continue

        # Start face recognition in a separate thread
        threading.Thread(target=recognize_faces, args=(frame.copy(),), daemon=True).start()

        # Break the loop if a face is detected and saved
        if known_face_encodings:
            break

        # Display the frame
        cv2.imshow('Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the frame processing function in the main thread
process_frames()

# Release the video capture object
cap.release()
