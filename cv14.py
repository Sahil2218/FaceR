import cv2
import numpy as np

# Load pre-trained pedestrian detection model (HOG)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open video capture (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Initialize variables
people_count = 0

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # If the frame read is successful
    if ret:
        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Detect people in the frame
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)

        # Draw bounding boxes around detected people
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update people count
        people_count = len(boxes)

        # Display the frame with bounding boxes
        cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('People Counting', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()