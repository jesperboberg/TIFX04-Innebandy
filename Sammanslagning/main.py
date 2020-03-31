import CentroidTracker
import numpy as np
import cv2
import pytesseract
import time
import math
import ctypes
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

team = "Pixbo"

# Coordinates for heat map
coordstemp = {}
coords = {}
totcoords = defaultdict(list)
totDist = defaultdict(int)

# Variables for sdpee speed over time

Running = defaultdict(int)
Jogging = defaultdict(int)
Walking = defaultdict(int)

Runningtemp = {}
Joggingtemp = {}
Walkingtemp = {}

X = {}
Y = {}

global error
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
xpoint = {}
ypoint = {}

xdist = {}
ydist = {}
absolutedist = {}

xdisttemp = {}
ydisttemp = {}
absolutedisttemp = {}

Sekund = 0
Minut = 0

xcoord = {}
ycoord = {}

boxes=np.zeros(4)
#val = 0 #Unnecessary?
#plotvals = [] #Unnecessary
throwvalue = 0

data = None

ct = CentroidTracker.CentroidTracker()

with open('input.json') as json_file:
    data = json.load(json_file)

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
                image = clone.copy()

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

        # If there are two reference points, then crop the region of interest
        # from teh image and display it
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

# Loop for choosing coordinates for projecting
while True:
    ret, framefield = field.read()
    if not ret:
        continue
    framefield = cv2.resize(framefield, (640, 480))
    cv2.imshow("test", framefield)
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

points = np.array(posList)

# Test i hallen hemma nedre vänster sen klockans varv o sist mitten. Punkter i planet man vill konvertera ner i.
pts_dst = np.array([[0,0], [0,200], [170,200], [170,0],[85,100]])

# Calculate matrix H for coordinate projecting to 2D
htransf, status = cv2.findHomography(points, pts_dst)

# Reset webcams
clock.release()
field.release()

clockimages = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
field = cv2.VideoCapture("jesper4.mov")

frame_width = int( field.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( field.get( cv2.CAP_PROP_FRAME_HEIGHT))

# Forground, background
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

def nothing(x):
    pass

ret1, frame1 = field.read()
ret2, frame2 = field.read()

start_time = time.time()

cv2.namedWindow('Control')

cv2.namedWindow("Final")
cv2.createTrackbar("Lower", "Final", 20,50,nothing)
cv2.createTrackbar("Higher", "Final", 255,255,nothing)
cv2.createTrackbar("D. Iterations", "Final", 20, 100, nothing)

i = 0
# Main loop
while field.isOpened():

    # Check if any frame is None then break
    if not ret2 or not ret1:
        print('continue bc: ', ret1, ret2)
        frame1 = frame2
        ret1 = ret2
        #reading a new value, this way the while loop will work
        ret2, frame2 = field.read()
        continue

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
    
    low = cv2.getTrackbarPos("Lower", "Final")
    high = cv2.getTrackbarPos("Higher", "Final")
    dilated = cv2.getTrackbarPos("D. Iterations", "Final")
    # Adjust threshold values here
    _, thresh = cv2.threshold(blur, low, high, cv2.THRESH_BINARY)

    #Obs that you can change iterations for refinement
    dilated = cv2.dilate(thresh, None, iterations=dilated)

    #Finding contours, obs this is a list/array
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate over all identified contours
    for contour in contours:
        #x,y coordinate, then width and height
        (x, y, w, h) = cv2.boundingRect(contour)
        boxes = np.array([x, y, x+w, y+h])
        #print(boxes)
    
        rect = [x,y,x+w,y+h] 
        rectList.append(rect)

        # Adjust how small boxes we tolerate here
        if cv2.contourArea(contour) < 40:
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    k = cv2.waitKey(1)

    objects = ct.update(rectList)
    rectList=[] #Detta var ändringen för ID grejen!!
    #print(objects)
    # Iterate over all detected objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #if objectID<4:
        #print(centroid)
        text = "ID {}".format(objectID)       
        cv2.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
        a = np.array([[centroid[0],centroid[1]]],dtype="float32")
        #a = np.array([[(boxes[0]+boxes[2])/2,(boxes[1]+boxes[3])/2]], dtype="float32")
        a = np.array([a]) #Convert from [[625. 178]] --> [[[625. 178]]]

        # TODO: this should be commented by someone who knows better than me
        aprim = cv2.perspectiveTransform(a, htransf)
        
        #xcoord = aprim[0][0][0]
        #ycoord = aprim[0][0][1]
        xcoord[objectID] = aprim[0][0][0]
        ycoord[objectID] = aprim[0][0][1]
        print("ID", objectID, "X ", xcoord[objectID], "Y ", ycoord[objectID])
        
        #x = abs(xpoint-aprim[0][0][0])
        #y = abs(ypoint-aprim[0][0][1]) 
        if objectID not in xpoint:
            xpoint[objectID] = xcoord[objectID] #annars 0
            ypoint[objectID] = ycoord[objectID]
        
        #x[objectID] = abs(xpoint-aprim[0][0][0])
        #y[objectID] = abs(ypoint-aprim[0][0][1]) 
        X[objectID] = abs(xpoint[objectID]-xcoord[objectID])
        Y[objectID] = abs(ypoint[objectID]-ycoord[objectID])
        if objectID in coordstemp:
            #coordstemp[objectID].append([xcoord, ycoord])
            coordstemp[objectID].append([xcoord[objectID], ycoord[objectID]])
        else:
            #coordstemp[objectID] = [[xcoord, ycoord]]
            coordstemp[objectID] = [[xcoord[objectID], ycoord[objectID]]]            
        #print(coordstemp)
        if objectID not in xdisttemp:
            xdisttemp[objectID] = 0
            ydisttemp[objectID] = 0
        else:
            #xdisttemp[objectID] = xdisttemp[objectID] + x
            #ydisttemp[objectID] = ydisttemp[objectID] + y
            xdisttemp[objectID] = xdisttemp[objectID] + X[objectID]
            ydisttemp[objectID] = ydisttemp[objectID] + Y[objectID]
        
        #print("ID: ", objectID, "X: ",X[objectID]) ##################################################################################################        
        #print("x",X,"xdisttemp",xdisttemp) #################################################
        #nowdistance = math.sqrt(x*x+y*y)
        nowdistance = math.sqrt(X[objectID]*X[objectID]+Y[objectID]*Y[objectID])

        if objectID in absolutedisttemp:
            absolutedisttemp[objectID] = absolutedisttemp[objectID] + nowdistance
        else:
            absolutedisttemp[objectID] = 0
        
        #xpoint = aprim[0][0][0]
        #ypoint = aprim[0][0][1]
        xpoint[objectID] = aprim[0][0][0]
        ypoint[objectID] = aprim[0][0][1]

        # Running faster than 2.68224 m/s, jogging between that and 1.38582 m/s. walking below that
        if nowdistance > Runningborder:
            if objectID is not None and objectID not in Runningtemp:
                Runningtemp[objectID] = 1
            else:
                Runningtemp[objectID] = Runningtemp[objectID] + 1
        elif nowdistance < Joggingborder:
            if objectID is not None and objectID not in Joggingtemp:
                Joggingtemp[objectID] = 1
            else:
                Joggingtemp[objectID] = Joggingtemp[objectID] + 1
        else:
            if objectID is not None and objectID not in Walkingtemp:
                Walkingtemp[objectID] = 1
            else:
                Walkingtemp[objectID] = Walkingtemp[objectID] + 1

    if (i % fps) == 0:
        ret, clock = clockimages.read()
        
        print("--- %s seconds ---" % (time.time() - start_time))
        small_frame = clock[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
        cv2.imshow("Small Frame", small_frame)
        error = mse(Ref_img, small_frame)

        Ref_img = small_frame
        
        if error < 0:
            print("Same image")
            coordstemp = {}
            absolutedisttemp = {}
            Runningtemp = {}
            Walkingtemp = {}
            Joggingtemp = {}
            xdisttemp = {}
            ydisttemp = {}
            throwvalue = throwvalue + 1
        else:
            Sekund = Sekund + 1
            if Sekund == 60:
                Sekund = 0
                Minut = Minut + 1

            for objectID in objects:
                
                if nowdistance > Runningborder:
                    if objectID is not None and objectID not in Runningtemp:
                        Runningtemp[objectID] = 1
                    else:
                        Runningtemp[objectID] = Runningtemp[objectID] + 1
                elif nowdistance < Joggingborder:
                    if objectID is not None and objectID not in Joggingtemp:
                        Joggingtemp[objectID] = 1
                    else:
                        Joggingtemp[objectID] = Joggingtemp[objectID] + 1
                else:
                    if objectID is not None and objectID not in Walkingtemp:
                        Walkingtemp[objectID] = 1
                    else:
                        Walkingtemp[objectID] = Walkingtemp[objectID] + 1
                
                if objectID in totcoords:
                    tmp = totcoords[objectID]
                    tmp.append(coordstemp[objectID])                    
                    totcoords[objectID] = tmp
                else:
                    totcoords[objectID] = coordstemp[objectID]

                val = 0
                if objectID in absolutedist:
                    xdist[objectID] = xdist[objectID] + xdisttemp[objectID]
                    ydist[objectID] = ydist[objectID] + ydisttemp[objectID]
                    absolutedist[objectID] = absolutedist[objectID] + absolutedisttemp[objectID]
                else:
                    xdist[objectID] = xdisttemp[objectID]
                    ydist[objectID] = ydisttemp[objectID]
                    absolutedist[objectID] = absolutedisttemp[objectID]
                if objectID < 3:
                    data["teams"][0]["players"][objectID]['distance'] = absolutedist[objectID]

                print("Absolute distance traveled for ID %d is: %.2f dm" % (objectID, absolutedist[objectID]))
            with open('input.json', 'w') as outfile:
                json.dump(data,outfile)
            
            absolutedisttemp = {}
            Runningtemp = {}
            Walkingtemp = {}
            Joggingtemp = {}
            xdisttemp = {}
            ydisttemp = {}
            coordstemp = {}
    
        #print(rect)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        

    image = cv2.resize(frame1, (640,480))
    dilated = cv2.resize(dilated, (640,480))
    dilated = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    subset2 = cv2.hconcat([image, dilated])
    cv2.imshow("Final", subset2)
    frame1 = frame2
    #reading a new value, this way the while loop will work
    ret2, frame2 = field.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if i == 950:
        break
    
# Exit GUI and release video streams
cv2.destroyAllWindows()
field.release()
clockimages.release()

for objectID in objects:
    # Distance calculations
    totDist[objectID] = Running[objectID] + Jogging[objectID] + Walking[objectID]
    if totDist[objectID] != 0:
        ProcentRunning[objectID] = 100*Running[objectID]/totDist[objectID]
        ProcentJogging[objectID] = 100*Jogging[objectID]/totDist[objectID]
        ProcentWalking[objectID] = 100*Walking[objectID]/totDist[objectID]

# Presenting results below
print("Längden av totcoords är: %.2f" % len(totcoords))
print('Iterations in mainloop: ', i)
print('Length of object list', len(objects))
#heatmap, xedges, yedges = np.histogram2d(*zip(*totcoords), bins=(64, 64), range=[[-30,30],[-30,30]]) # Bins sätter upplösningen
# Range sätter faktiska begränsningar. Notera att y-axeln är inverterad i grafen
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#plt.clf()
#plt.title("Heatmap over player movement")
#plt.ylabel("Kortsida")
#plt.xlabel("Långsida")
#plt.gca().invert_yaxis()
#plt.imshow(heatmap, extent=extent, origin="lower")
#plt.show()

#plt.plot(*zip(*totcoords))
#plt.suptitle("Rörelse över planen")
#plt.axis([0, 30, 0, 30])
#plt.show()

with open("input.json") as json_file:
    parsed = json.load(json_file)
print(json.dumps(parsed, indent=4))

for objectID in objects:
    print('STATISTICS FOR ID %d' % objectID)
    if objectID in xdist:
        print("X-distance traveled is: %.2f dm" % xdist[objectID])
    if objectID in ydist:
        print("Y-distance traveled is: %.2f dm" % ydist[objectID])
    if objectID in absolutedist:
        print("Absolute distance traveled is: %.2f dm" % absolutedist[objectID])
    if objectID not in xdist and objectID not in ydist and objectID not in absolutedist:
        print('no data collected')
    print('------------------------------------------')

#print("Procent running: %.2f procent" % ProcentRunning)
#print("Procent jogging: %.2f procent" % ProcentJogging)
#print("Procent walking: %.2f procent" % ProcentWalking)

print("Antalet kastade gånger = %.2f" % throwvalue)
