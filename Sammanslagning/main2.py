'''
This code will work differently depending on different videofiles:

*Consider changing how small rectangles we tolerate - line 319
*Consider increasing the iterations - Line 302
'''

import CentroidTracker
import numpy as np
import cv2
import pytesseract
import time
import math
import ctypes
#import Image
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from anaconda_project import status

# TODO: Backen behöver du denna linen?
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

# Coordinates for heat map
coordstemp = []
coords = []
totcoords = []

# Variables for sdpee speed over time
Running = 0
Jogging = 0
Walking = 0
Runningtemp = 0
Joggingtemp = 0
Walkingtemp = 0

x = 0
y = 0

global error #global keyword allows you to modify the variable outside of the current scope.
error = 0
i = 0

#Constants
fps = 30
Runningborder = 26.8224/fps
Joggingborder = 13.8582/fps

img_counter = 0

posList = []
click = 0
g = 0

# Storing rects to track them later on
rectList = []

# Unsorted variables
xpoint = 0
ypoint = 0

xdist = 0
ydist = 0
absolutedist = 0

xdisttemp = 0
ydisttemp = 0
absolutedisttemp = 0

Sekund = 0
Minut = 0

xcoord = 0
ycoord = 0

boxes=np.zeros(4)
val = 0
plotvals = []
throwvalue = 0

# PROGRAM SETUP

# Method for choosing area for clock
def shape_selection(event, x, y, flags, param):
    # grab references to the global variables
    global ref_point, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        ref_point.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(frameclock, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", frameclock)

# Mean square error function
def mse(imageA, imageB):
    err = 0
    # The "MeanSquaredError" between the two images is the sum of the squared difference between the two images:
    # TODO: Detta är fult som fan
    if imageA is not None:
        if imageB is not None:
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])
    # Return the mse, the lower the error, the more "similar" the two images are
    return err

# Method for choosing coordinates to project
def onMouse(event, i, j, flags, param):
    global posList
    global click
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((i, j))
        click = click + 1
        cv2.circle(framefield, (i, j), 2, (0, 255, 0), 2)
        cv2.putText(framefield, "{}".format(click), (i+3, j+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(img_name, framefield)

clock = cv2.VideoCapture(0)
cv2.namedWindow("test")

# Running while choosing clock area
while True:
    ret, frameclock = clock.read()
    frameclock = cv2.resize(frameclock, (640, 480))
    frameclock = cv2.cvtColor(frameclock, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", frameclock)
    
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 32:
        img_counter +=1
        clone = frameclock.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", shape_selection) # Press space to choose the frame you want to extract

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv2.imshow("image", frameclock)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                print("hej")
                image = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        # If there are two reference points, then crop the region of interest
        # from the image and display it
        if len(ref_point) == 2:
            crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            
            cv2.imshow("crop_img", crop_img)
            cv2.imwrite("Crop_img.jpg", crop_img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    if k%256 == 27:
        # Esc pressed
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break

# TODO: This shouldn't be saved on computer, faster in cache somehow
Ref_img = cv2.imread("crop_img.jpg")
#Ref_img = crop_img

cv2.startWindowThread()
cv2.namedWindow("test")
field = cv2.VideoCapture("jesper4.mov")

# Loop for choosing coordinates for projection
while True:
    ret, framefield = field.read()
    framefield = cv2.resize(framefield, (640, 480))
    cv2.imshow("test", framefield)
    if not ret:
        break
    k = cv2.waitKey(1)

    # Press ESC to close windows and continue
    if k%256 == 27:
        # Esc pressed
        print("Escape hit, closing...")
        cv2.destroyAllWindows()
        break

    # Press r to reset posList and re-choose coordinates
    if k%256 == ord("r"):
        posList = []

    # Press c to see chosen coordinates
    if k%256 == ord("c"):

        for n in posList:
            g = g+1
            cv2.circle(framefield, n, 2, (0, 255, 0), 2)
            cv2.putText(framefield, "{}".format(g), n, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(img_name, framefield)

    # Press space to take screenshot
    elif k%256 == 32:
        # SPACE pressed
        img_name = "WindowName"
        cv2.imshow(img_name, framefield)
        print("{} written!".format(img_name))
        img_counter +=1

        cv2.setMouseCallback('WindowName', onMouse)
        # Convert to numpy for other usages
        posNp = np.array(posList)    

#TODO: unneccessary convertation 
print(posList)
points = np.array(posList)

# Test i hallen hemma nedre vänster sen klockans varv o sist mitten. Punkter i planet man vill konvertera ner i.
pts_dst = np.array([[0,0], [0,40], [40,40], [40,0],[20,20]])

# Calculate matrix H for coordinate projecting to 2D, status = [[1] [1] [1] [1] [1]]
htransf, status = cv2.findHomography(points, pts_dst)

# Reset webcams
clock.release()
field.release()

print(htransf)

clockimages = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
field = cv2.VideoCapture("jesper4.mov")

frame_width = int( field.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int( field.get( cv2.CAP_PROP_FRAME_HEIGHT))

# Forground, background, Note: does nothing here
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

def nothing(x):
    pass

ct = CentroidTracker.CentroidTracker()

ret, frame1 = field.read()
ret2, frame2 = field.read()

start_time = time.time()

# Mainloop
while field.isOpened():

    # Check if any frame is None then break
    if frame2 is None:
        print("Frame stream broken")
        break

    if frame1 is None:
        print("Frame stream broken")
        break

    i = i+1

    #Difference between first and second frame
    diff = cv2.absdiff(frame1, frame2)
    #Convert difference to gray scale mode, easier to use
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #Smoothing the picture
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adjust threshold values here
    _, thresh = cv2.threshold(blur, 27, 200, cv2.THRESH_BINARY)

    #Obs that you can change iterations for refinement
    dilated = cv2.dilate(thresh, None, iterations=8)

    #Finding contours, obs this is a list/array
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over all identified contours
    for contour in contours:
        #x,y coordinate, then width and height
        (x, y, w, h) = cv2.boundingRect(contour)
        boxes = np.array([x, y, x+w, y+h])

        rect = [x,y,x+w,y+h] 
        rectList.append(rect)
        #print(rect)
        #print(rectList)

        # Adjust how small boxes we tolerate here
        if cv2.contourArea(contour) < 40:
            continue

        # Adjust how big boxes we tolerate here
        #if cv2.contourArea(contour) > 10000:
        #   continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        a = np.array([[boxes[0],boxes[1]]], dtype="float32")
        # TODO: next line shouldn't matter right?
        a = np.array([a])

        # TODO: this should be commented by someone who knows better than me
        aprim = cv2.perspectiveTransform(a, htransf)
        xcoord = aprim[0][0][0]
        ycoord = aprim[0][0][1]
        x = abs(xpoint-aprim[0][0][0])
        y = abs(ypoint-aprim[0][0][1])
        coordstemp.append((xcoord, ycoord))
        #print(coordstemp)
        xdisttemp = xdisttemp + x
        ydisttemp = ydisttemp + y
        nowdistance = math.sqrt(x*x+y*y)
        absolutedisttemp = absolutedisttemp + nowdistance
        xpoint = aprim[0][0][0]
        ypoint = aprim[0][0][1]

        # Running faster than 2.68224 m/s, jogging between that and 1.38582 m/s. walking below that
        if nowdistance > Runningborder:
            Runningtemp = Runningtemp + 1

        elif nowdistance < Joggingborder:
            Walkingtemp = Walkingtemp + 1
        else:
            Joggingtemp = Joggingtemp + 1

    k = cv2.waitKey(1)

    #objects = ct.update(rect)
    objects = ct.update(rectList)
    rectList=[] #Detta var ändringen för ID grejen!!

    # Iterate over all detected objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #if objectID<4:
            text = "ID {}".format(objectID)
        #print(objectID)        
            cv2.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
        #cv2.putText(fgmask, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.circle(fgmask, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    if (i % fps) == 0:
        ret, clock = clockimages.read()
        print(ret)
        #print(clock[ref_point[0][1]]:ref_point[1][1])
        
        print("--- %s seconds ---" % (time.time() - start_time))
        small_frame = clock[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        cv2.imshow("Small Frame", small_frame)
        error = mse(Ref_img, small_frame)

        Ref_img = small_frame
        
        if error < 100:
            print("Same image")
            #print("%.2f : %.2f" % (Minut, Sekund))
            #cv2.imshow("Framefield", framefield)
            coordstemp = []
            absolutedisttemp = 0
            Runningtemp = 0
            Walkingtemp = 0
            Joggingtemp = 0
            xdisttemp = 0
            ydisttemp = 0
            throwvalue = throwvalue + 1
        else:
            #print("New image")
            Sekund = Sekund + 1
            if Sekund == 60:
                Sekund = 0
                Minut = Minut + 1
            #print("%.2f : %.2f" % (Minut, Sekund))
            #print("Här tas mätningar")
            Jogging = Jogging + Joggingtemp
            Running = Running + Runningtemp
            Walking = Walking + Walkingtemp
            totcoords.extend(coordstemp)
            #print(coordstemp)
            val = 0
            coordstemp = []

            xdist = xdist + xdisttemp
            ydist = ydist + ydisttemp
            absolutedist = absolutedist + absolutedisttemp
            #print("X-distance traveled is: %.2f dm" % xdist)
            #print("Y-distance traveled is: %.2f dm" % ydist)
            print("Absolute distance traveled is: %.2f dm" % absolutedist)
            absolutedisttemp = 0
            Runningtemp = 0
            Walkingtemp = 0
            Joggingtemp = 0
            xdisttemp = 0
            ydisttemp = 0
        #print(rect)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        

    image = cv2.resize(frame1, (640,480))
    #out.write(image)
    cv2.imshow("feed", image)
    #cv2.imshow("feed2", fgmask)
    frame1 = frame2
    #reading a new value, this way the while loop will work
    ret3, frame2 = field.read()
    #print("ret3 = ")
    #print(ret3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Exit GUI and release video streams
cv2.destroyAllWindows()
field.release()
clockimages.release()

# Distance calculations
totDist = Running + Jogging + Walking
if totDist != 0:
    ProcentRunning = 100*Running/totDist
    ProcentJogging = 100*Jogging/totDist
    ProcentWalking = 100*Walking/totDist

# Presenting results below
print(totcoords)
print("Längden av totcoords är: %.2f" % len(totcoords))

heatmap, xedges, yedges = np.histogram2d(*zip(*totcoords), bins=(64, 64), range=[[-50,150],[-50,150]]) # Bins sätter upplösningen
# Range sätter faktiska begränsningar. Notera att y-axeln är inverterad i grafen
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.title("Heatmap over player movement")
plt.ylabel("Kortsida")
plt.xlabel("Långsida")
plt.gca().invert_yaxis()
plt.imshow(heatmap, extent=extent, origin="lower")
plt.show()

#plt.plot(*zip(*totcoords))
#plt.suptitle("Rörelse över planen")
#plt.axis([0, 30, 0, 30])
#plt.show()


print("X-distance traveled is: %.2f dm" % xdist)
print("Y-distance traveled is: %.2f dm" % ydist)
print("Absolute distance traveled is: %.2f dm" % absolutedist)

print("Procent running: %.2f procent" % ProcentRunning)
print("Procent jogging: %.2f procent" % ProcentJogging)
print("Procent walking: %.2f procent" % ProcentWalking)

print("Antalet kastade gånger = %.2f" % throwvalue)
