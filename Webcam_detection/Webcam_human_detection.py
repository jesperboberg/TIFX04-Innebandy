import numpy as np
import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
cap = cv2.VideoCapture(1)

# The output will be written to webcam_output.avi
out = cv2.VideoWriter("webcam_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15., (640, 480))


while(True):
    # Reading the frame
    ret, frame = cap.read()

    # Resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # Using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect people in the image, returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))

    boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # Display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Write the output video
    out.write(frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
# And release the output
out.release()
# Finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
