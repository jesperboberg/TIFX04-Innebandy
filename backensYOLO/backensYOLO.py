from __future__ import division
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
from ftplib import FTP 
import os
import fileinput
import torch 
import torch.nn as nn
from torch.autograd import Variable 
from util import *
import argparse 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import tensorflow as tf

# Load YOLO network

images = "imgs"
batch_size = 1
confidence = 0.5
nms_thesh = 0.4
start = 0
CUDA = False

num_classes = 80
classes = load_classes("data/coco.names")

#Set up the neural network
print("Loading network.....")
model = Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
print("Network successfully loaded")

model.net_info["height"] = "416"
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

# Set model in evaluation mode


# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False

team = "Pixbo"

# Coordinates for heat map
coordstemp = defaultdict(list) # Var (list) förrut
coords = defaultdict(int)
totcoords = defaultdict(list) # Var (list) förrut
totDist = defaultdict(int)

# Variables for sdpee speed over time

Running = defaultdict(int)
Jogging = defaultdict(int)
Walking = defaultdict(int)

Runningtemp = defaultdict(int)
Joggingtemp = defaultdict(int)
Walkingtemp = defaultdict(int)

ProcentRunning = defaultdict(int)
ProcentJogging = defaultdict(int)
ProcentWalking = defaultdict(int)

x = 0
y = 0

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

xdist = defaultdict(int)
ydist = defaultdict(int)
absolutedist = defaultdict(int)

xdisttemp = defaultdict(int)
ydisttemp = defaultdict(int)
absolutedisttemp = defaultdict(int)

Sekund = 0
Minut = 0

xcoord = 0
ycoord = 0

boxes=np.zeros(4)
val = 0
plotvals = []
throwvalue = 0

tmp = []

data = None
# Test i hallen hemma nedre vänster sen klockans varv o sist mitten. Punkter i planet man vill konvertera ner i.
#pts_dst = np.array([[0,0], [0,200], [170,200], [170,0],[85,100]])
pts_dst = np.array([[0,0], [61,0], [61,67], [0,67], [30.5,30]])

ct = CentroidTracker.CentroidTracker()

#ftp = FTP()
#ftp.set_debuglevel(2)
#ftp.connect('ftp.hindret.eu', 21) 
#ftp.login('hindret.eu','TIFXib2020')
#ftp.cwd('/IB')

with open('/home/gustav/TIFX04/Arturs_kod/input.json') as json_file:
    data = json.load(json_file)

def tensor_conv(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return tf.matmul(arg, arg) + arg

def ftp_upload(localfile, remotefile):
  fp = open(localfile, 'rb')
  try:
    ftp.storbinary('STOR %s' % remotefile, fp, 1024)
  except Exception:
    print("remotefile not exist error caught" + remotefile)
    path,filename = os.path.split(remotefile)
    print("creating directory: " + remotefile)
    ftp.mkd(path)
    ftp_upload(localfile, remotefile)
    fp.close()
    return
  fp.close()
  print ("after upload " + localfile + " to " + remotefile)

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

# Loop for choosing coordinates for projecting
while True:
    field = cv2.VideoCapture("/home/gustav/TIFX04/Arturs_kod/test3.mp4")
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

# Calculate matrix H for coordinate projecting to 2D
htransf, status = cv2.findHomography(points, pts_dst)

# Reset webcams
clock.release()
field.release()

clockimages = cv2.VideoCapture(0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
field = cv2.VideoCapture("/home/gustav/TIFX04/Arturs_kod/test2.mp4")

frame_width = int( field.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( field.get( cv2.CAP_PROP_FRAME_HEIGHT))

# Forground, background
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

def nothing(x):
    pass



start_time = time.time()

cv2.namedWindow("Final")

k=0
i = 0
# Main loop
while field.isOpened():

    ret1, frame1 = field.read()

    if i%5 == 0:
        ret1, frame1 = field.read()

        # Check if any frame is None then break
        if not ret1:
            k = k+1
            if k == 7:
                break
            continue


        # Check if the frame is None then break
        if frame1 is None:
            print("Frame stream broken")
            break

        batch = prep_image(frame1, inp_dim)
        im_dim = (frame1.shape[1], frame1.shape[0])
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)
        
        prediction = model(Variable(batch), False)

        prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    
        objs = [classes[int(x[-1])] for x in prediction]

        k = cv2.waitKey(1)

        im_dim = torch.index_select(im_dim, 0, prediction[:,0].long())

        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)


        prediction[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        prediction[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2



        prediction[:,1:5] /= scaling_factor

        for i in range(prediction.shape[0]):
            prediction[i, [1,3]] = torch.clamp(prediction[i, [1,3]], 0.0, im_dim[i,0])
            prediction[i, [2,4]] = torch.clamp(prediction[i, [2,4]], 0.0, im_dim[i,1])

        for objects in prediction:
            if objects[0] == 0:
                rectList.append([float(objects[1]),float(objects[2]), float(objects[3]), float(objects[4])])
                cv2.rectangle(frame1, (int(objects[1]),int(objects[2])), (int(objects[3]), int(objects[4])), (0, 255, 0), 2)
                
        #objects = ct.update(rect)
        objects = ct.update(rectList)
        rectList=[] #Detta var ändringen för ID grejen!!

        # Iterate over all detected objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            #if objectID<4:
        
            text = "ID {}".format(objectID)
            cv2.putText(frame1, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            a = np.array([[centroid[0],centroid[1]]], dtype="float32")
            # TODO: next line shouldn't matter right?
            a = np.array([a])

            # TODO: this should be commented by someone who knows better than me
            aprim = cv2.perspectiveTransform(a, htransf)
            #if objectID == 0:
                #print(aprim[0][0])
            xcoord = aprim[0][0][0]
            ycoord = aprim[0][0][1]
            try:
                x = abs(xpoint[objectID]-aprim[0][0][0])
                y = abs(ypoint[objectID]-aprim[0][0][1])
                if objectID in coordstemp:
                    coordstemp[objectID].append((xcoord, ycoord))
                else:
                    coordstemp[objectID] = [(xcoord, ycoord)]
                if objectID not in xdisttemp:
                    xdisttemp[objectID] = 0
                    ydisttemp[objectID] = 0
                else:
                    xdisttemp[objectID] = xdisttemp[objectID] + x
                    ydisttemp[objectID] = ydisttemp[objectID] + y


                nowdistance = math.sqrt(x*x+y*y)

                if objectID in absolutedisttemp:
                    absolutedisttemp[objectID] = absolutedisttemp[objectID] + nowdistance
                else:
                    absolutedisttemp[objectID] = 0

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
                        Walkingtemp[objectID] = 1
                    else:
                        Walkingtemp[objectID] = Walkingtemp[objectID] + 1
                else:
                    if objectID is not None and objectID not in Walkingtemp:
                        Joggingtemp[objectID] = 1
                    else:
                        Joggingtemp[objectID] = Joggingtemp[objectID] + 1
            except KeyError:
                xpoint[objectID] = aprim[0][0][0]
                ypoint[objectID] = aprim[0][0][1]


        if (i % fps) == 0:
            ret, clock = clockimages.read()
            #print(clock[ref_point[0][1]]:ref_point[1][1])

            print("--- %s seconds ---" % (time.time() - start_time))
            small_frame = clock[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("Small Frame", small_frame)
            error = mse(Ref_img, small_frame)

            Ref_img = small_frame

            if error < 0:
                print("Same image")
                #print("%.2f : %.2f" % (Minut, Sekund))
                #cv2.imshow("Framefield", framefield)
                coordstemp = {}
                absolutedisttemp = {}
                Runningtemp = {}
                Walkingtemp = {}
                Joggingtemp = {}
                xdisttemp = {}
                ydisttemp = {}
                throwvalue = throwvalue + 1
            else:
                print("New image")
                Sekund = Sekund + 1
                if Sekund == 60:
                    Sekund = 0
                    Minut = Minut + 1
                #print("%.2f : %.2f" % (Minut, Sekund))
                #print("Här tas mätningar")

                for objectID in objects:
                    print("objectID: ")
                    print(objectID)

                    if objectID is not None and objectID in Runningtemp:
                        Running[objectID] = Running[objectID] + Runningtemp[objectID]
                    else:
                        Running[objectID] = Running[objectID]

                    if objectID is not None and objectID in Joggingtemp:
                        Jogging[objectID] = Jogging[objectID] + Joggingtemp[objectID]
                    else:
                        Jogging[objectID] = Jogging[objectID]

                    if objectID is not None and objectID in Walkingtemp:
                        Walking[objectID] = Walking[objectID] + Walkingtemp[objectID]
                    else:
                        Walking[objectID] = Walking[objectID]

                    Walkingtemp[objectID] = 0
                    Joggingtemp[objectID] = 0
                    Runningtemp[objectID] = 0

                    #print("Totcoords: ")
                    #print(totcoords)
                    #print("objectID: ")
                    #print(objectID)

                    if objectID is not None and objectID in coordstemp:
                        print("ObjectID i totcoords")
                        tmp = totcoords[objectID]
                        tmp.extend(coordstemp[objectID])
                        totcoords[objectID] = tmp
                        coordstemp[objectID] = []
                    else:
                        print("totcoords[objectID: ")
                        print(totcoords[objectID])

                        print("coordstemp[objectID: ")
                        print(coordstemp[objectID])
                        totcoords[objectID] = coordstemp[objectID]
                        coordstemp[objectID] = []
                        print("Totcoords är:")
                        print(totcoords)
                    #print(coordstemp)
                    val = 0
                    if objectID is not None and objectID in xdisttemp:
                        print("ObjectID i absolutedist")
                        print(xdist[objectID])
                        xdist[objectID] = xdist[objectID] + xdisttemp[objectID]
                        ydist[objectID] = ydist[objectID] + ydisttemp[objectID]
                        absolutedist[objectID] = absolutedist[objectID] + absolutedisttemp[objectID]
                    else:
                        xdist[objectID] = xdisttemp[objectID]
                        ydist[objectID] = ydisttemp[objectID]
                        absolutedist[objectID] = absolutedisttemp[objectID]
                    if objectID < 3:
                        data["teams"][0]["players"][objectID]['distance'] = absolutedist[objectID]
                #print("X-distance traveled is: %.2f dm" % xdist)
                #print("Y-distance traveled is: %.2f dm" % ydist)
                    print("Absolute distance traveled for ID %d is: %.2f dm" % (objectID, absolutedist[objectID]))
                with open('/home/gustav/TIFX04/Arturs_kod/input.json', 'w') as outfile:
                    json.dump(data,outfile)
                #ftp_upload('/home/gustav/TIFX04/Arturs_kod/input.json', 'input.json')
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
        cv2.imshow("Final", image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    i += 1

# Exit GUI and release video streams
cv2.destroyAllWindows()
field.release()
clockimages.release()
#ftp_upload('/home/gustav/TIFX04/Arturs_kod/input.json', 'input.json')
#ftp.quit()

for objectID in objects:
    # Distance calculations
    totDist[objectID] = Running[objectID] + Jogging[objectID] + Walking[objectID]
    if totDist[objectID] != 0:
        ProcentRunning[objectID] = 100*Running[objectID]/totDist[objectID]
        ProcentJogging[objectID] = 100*Jogging[objectID]/totDist[objectID]
        ProcentWalking[objectID] = 100*Walking[objectID]/totDist[objectID]

# Presenting results below

#print(totDist)
#print("---------------------------------------------------------------------------")
#
#print(totcoords)

print("Längden av totcoords är: %.2f" % len(totcoords))
print('Iterations in mainloop: ', i)
print('Length of object list', len(objects))
#heatmap, xedges, yedges = np.histogram2d(*zip(*totcoords[1]), bins=(200, 200), range=[[200,300],[-70,-30]]) # Bins sätter upplösningen
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

with open("/home/gustav/TIFX04/Arturs_kod/input.json") as json_file:
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
    if objectID in ProcentRunning:
        print("Percentage of travel as running: %.2f" % ProcentRunning[objectID])
    if objectID in ProcentJogging:
        print("Percentage of travel as jogging: %.2f" % ProcentJogging[objectID])
    if objectID in ProcentWalking:
        print("Percentage of travel as walking: %.2f" % ProcentWalking[objectID])
    if objectID not in xdist and objectID not in ydist and objectID not in absolutedist:
        print('no data collected')
    print('------------------------------------------')

#print("Procent running: %.2f procent" % ProcentRunning)
#print("Procent jogging: %.2f procent" % ProcentJogging)
#print("Procent walking: %.2f procent" % ProcentWalking)

print("Antalet kastade gånger = %.2f" % throwvalue)
