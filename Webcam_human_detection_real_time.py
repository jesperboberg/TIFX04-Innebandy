import numpy as np
import cv2
import math
import winsound

def onMouse(event, i, j, flags, param):
    global posList
    global click
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((i, j))
        click = click + 1
        cv2.circle(frame, (i, j), 2, (0, 255, 0), 2)
        cv2.putText(frame, "{}".format(click), (i+3, j+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(img_name, frame)


# 0 för vanlig webcamera, 1 för extern
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

posList = []
click = 0
g = 0
while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    # Tryck ESC och klicka ner alla videofönster för att fortsätta.
    if k%256 == 27:
        # Esc pressed
        print("Escape hit, closing...")
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        winsound.PlaySound("Ready_sound.wav", winsound.SND_FILENAME)
        break
    # Tryck r för att tömma listan över positioner.
    if k%256 == ord("r"):
        posList = []
    # Tryck c för att se var alla punkter hamnar.
    if k%256 == ord("c"):
        for n in posList:
            g = g+1
            cv2.circle(frame, n, 2, (0, 255, 0), 2)
            cv2.putText(frame, "{}".format(g), n, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(img_name, frame)
    # Tryck SPACE för att ta en screenshot av webcameran.
    elif k%256 == 32:
        # SPACE pressed
        img_name = "WindowName"
        cv2.imshow(img_name, frame)
        print("{} written!".format(img_name))
        img_counter +=1

        cv2.setMouseCallback('WindowName', onMouse)
        posNp = np.array(posList)     # convert to numpy for other usages
#cv2.waitKey(0)
#cv2.destroyAllWindows()
print(posList)
points = np.array(posList)

# Test i hallen hemma nedre vänster sen klockans varv o sist mitten. Punkter i planet man vill konvertera ner i.
pts_dst = np.array([[0,0], [0,30], [10,30], [10,0], [5,15]])

# Calculate matrix H
h, status = cv2.findHomography(points, pts_dst)




print(h)


# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
cap = cv2.VideoCapture(0)

# The output will be written to webcam_output.avi
#out = cv2.VideoWriter("webcam_output_coord_test1.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15., (640, 480))
coords = []
xpoint = 0
ypoint = 0

xdist = 0
ydist = 0
absolutedist = 0

fps = 10
Runningborder = 26.8224/fps
Joggingborder = 13.8582/fps

Running = 0
Jogging = 0
Walking = 0

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

    for (xA, yA, xB, yB) in boxes: # Assume 30 fps
        # Display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        a = np.array([[xA,yA]], dtype="float32")
        a = np.array([a])
        #print("a är:")
        #print(a)
        #print("New a")
        aprim = cv2.perspectiveTransform(a, h)
        coords.append(aprim)
        #print("xA, yA = {}".format([xA,yA]))
        #print("Primmade = {}".format(aprim))
        x = abs(xpoint-aprim[0][0][0])
        y = abs(ypoint-aprim[0][0][1])
        xdist = xdist + x
        ydist = ydist + y
        nowdistance = math.sqrt(x*x+y*y)
        absolutedist = absolutedist + nowdistance
        xpoint = aprim[0][0][0]
        ypoint = aprim[0][0][1]
        # Running faster than 2.68224 m/s, jogging between that and 1.38582 m/s. walking below that
        if nowdistance > Runningborder:
            Running = Running + 1

        elif nowdistance < Joggingborder:
            Walking = Walking + 1

        else:
            Jogging = Jogging + 1

    # Write the output video
    #out.write(frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
# And release the output
#out.release()
# Finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
cam.release()
#print(coords)
#print(len(coords))

ProcentRunning = 100*Running/(Running + Jogging + Walking)
ProcentJogging = 100*Jogging/(Running + Jogging + Walking)
ProcentWalking = 100*Walking/(Running + Jogging + Walking)

print("X-distance traveled is: {}".format(xdist))
print("Y-distance traveled is: {}".format(ydist))
print("Absolute distance traveled is: {}".format(absolutedist))

print("Procent running: {} %".format(ProcentRunning))
print("Procent jogging: {} %".format(ProcentJogging))
print("Procent walking: {} %".format(ProcentWalking))
