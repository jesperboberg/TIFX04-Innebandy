import numpy as np
import cv2
import pytesseract
import time
import  math
import ctypes
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_point = []
cropping = False


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

def mse(imageA, imageB):
    # The "MeanSquaredError" between the two images is the sum of the squared difference between the two images:
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # Return the mse, the lower the error, the more "similar" the two images are
    return err

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
img_counter = 0

prestart = ctypes.windll.user32.MessageBoxW(0, "Every time a message such as this is shown, please close the message "
                                               "before proceeding with the instructions relayed.", "Pre-Start", 1)
start = ctypes.windll.user32.MessageBoxW(0, "To start the calibration click 'SPACE'.", "Start", 1)

while True:
    ret, frameclock = clock.read()
    frameclock = cv2.resize(frameclock, (640, 480))
    frameclock = cv2.cvtColor(frameclock, cv2.COLOR_BGR2GRAY)
    cv2.imshow("test", frameclock)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 32:
        answerruta = ctypes.windll.user32.MessageBoxW(0, "Drag a box over the gameclock, when you're done hit c to check",
                                                      "Tutorial", 1)
        if answerruta == 2:
            break
        img_counter +=1
        clone = frameclock.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", shape_selection) # Tryck SPACE och dra sedan den formen man vill extracta


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
                checkruta = ctypes.windll.user32.MessageBoxW(0, "If you're satisfied with your area hit 'ESC'."
                                                                "If you want to change it, hit 'r' and the 'SPACE'.",
                                                             "Check box", 1)
                break

        # if there are two reference points, then crop the region of interest
        # from teh image and display it
        if len(ref_point) == 2:
            crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("crop_img", crop_img)
            cv2.waitKey(0)

        # close all open windows
        cv2.destroyAllWindows()
        cont = ctypes.windll.user32.MessageBoxW(0, "Press 'ESC' to continue", "Continue", 1)

    if k%256 == 27:
        # Esc pressed
        print("Escape hit, closing...")
        #cv2.waitKey(0)
        cv2.destroyAllWindows()

        break



Ref_img = crop_img

cv2.startWindowThread()

field = cv2.VideoCapture(1)
cv2.namedWindow("test")
posList = []
click = 0
g = 0

startpoints = ctypes.windll.user32.MessageBoxW(0, "Press 'SPACE' to continue.",1)


while True:
    ret, framefield = field.read()
    framefield = cv2.resize(framefield, (640, 480))
    cv2.imshow("test", framefield)
    if not ret:
        break
    k = cv2.waitKey(1)
    # Tryck ESC och klicka ner alla videofönster för att fortsätta.
    if k%256 == 27:
        # Esc pressed
        print("Escape hit, closing...")
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
    # Tryck r för att tömma listan över positioner.
    if k%256 == ord("r"):
        posList = []
    # Tryck c för att se var alla punkter hamnar.
    if k%256 == ord("c"):
        checkpoints = ctypes.windll.user32.MessageBoxW(0, "If you're satisfied with your points click 'ESC'. If you wish"
                                                       "to change them click 'r'. This will empty the list of points."
                                                       , "Check points", 1)
        for n in posList:
            g = g+1
            cv2.circle(framefield, n, 2, (0, 255, 0), 2)
            cv2.putText(framefield, "{}".format(g), n, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(img_name, framefield)
    # Tryck SPACE för att ta en screenshot av webcameran.
    elif k%256 == 32:
        answerpoints = ctypes.windll.user32.MessageBoxW(0, "Click first on the 4 free-stroke dots starting bottom left"
                                                           " and working your way around the field. After that click "
                                                           "first on the bottom and then the top of the two corners "
                                                           "closest to the field of the goaliearea starting with the "
                                                           "left and then the right. Finish by clicking on the spot in "
                                                           "the middle of the field. "
                                                           "In total you should have clicked on 9 points."
                                                           "When you have clicked on every point hit 'c' to check them",
                                                        "Tutorial", 1)
        if answerpoints == 2:
            break
        # SPACE pressed
        img_name = "WindowName"
        cv2.imshow(img_name, framefield)
        print("{} written!".format(img_name))
        img_counter +=1
        #print("Click first on the 4 free-stroke dots starting bottom left and working your way around the field. "
        #      "After that click first on the bottom and then the top of the two corners closest to the center of the "
        #      "field of the goaliearea starting with the left and then the right. Finish by clicking on the spot in the"
        #      " middle of the field. "
        #      "In total you should have clicked on 9 points.")

        cv2.setMouseCallback('WindowName', onMouse)
        posNp = np.array(posList)     # convert to numpy for other usages
#cv2.waitKey(0)
#cv2.destroyAllWindows()
print(posList)
points = np.array(posList)

# Test i hallen hemma nedre vänster sen klockans varv o sist mitten. Punkter i planet man vill konvertera ner i.
pts_dst = np.array([[0,0], [0,30], [10,30], [10,0], [5,15]])

# Click first on the 4 free-stroke dots starting bottom left and working your way around the field. After that click
# first on the bottom and then the top of the two corners of the goaliearea starting with the left and then the right.
# Finish by clicking on the spot in the middle of the field. In total you should have clicked on 9 points.
# pts_dst = np.array([[35,15], [35,185], [365,185], [365,15], [68.5,75], [68.5,125], [331.5,125], [331.5,125], [200,100]])

# Calculate matrix H
h, status = cv2.findHomography(points, pts_dst)

clock.release()
field.release()



print(h)

clock = cv2.VideoCapture(0)
Sekund = 0
Minut = 0

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam video stream 0 for built-in webcam, 1 for external webcam
field = cv2.VideoCapture(1)

#cv2.startWindowThread()
coordstemp = []
coords = []
totcoords = []
xpoint = 0
ypoint = 0

xdist = 0
ydist = 0
absolutedist = 0

xdisttemp = 0
ydisttemp = 0
absolutedisttemp = 0

fps = 15
Runningborder = 26.8224/fps
Joggingborder = 13.8582/fps

Running = 0
Jogging = 0
Walking = 0

Runningtemp = 0
Joggingtemp = 0
Walkingtemp = 0

x = 0
y = 0

global error
error = 0
i = 0

xcoordtemp = []
ycoordtemp = []

xcoord = []
ycoord = []

startmeasure = ctypes.windll.user32.MessageBoxW(0, "To start measurement press 'OK'. To finish the measurement and exit"
                                                   " the program press 'ESC' when you wish to exit.",
                                                "Start Measurement", 1)
val = 0
plotvals = []
start_time = time.time()
while True:
        i = i+1
        #print(i)
        #time.sleep(1)
        cv2.startWindowThread()
        retclock, frameclock = clock.read()

        retfield, framefield = field.read()

        framefield = cv2.resize(framefield, (640, 480))

        frameclock = cv2.resize(frameclock, (640, 480))

        frameclock = cv2.cvtColor(frameclock, cv2.COLOR_BGR2GRAY)

        boxes, weights = hog.detectMultiScale(framefield, winStride=(8,8))

        boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])


        for (xA, yA, xB, yB) in boxes: # Assume 30 fps
            # Display the detected boxes in the colour picture

            cv2.rectangle(framefield, (xA, yA), (xB, yB), (0, 255, 0), 2)
            a = np.array([[xA,yA]], dtype="float32")
            #print(a)
            a = np.array([a])
            aprim = cv2.perspectiveTransform(a, h)
            xcoordtemp = aprim[0]
            ycoordtemp = aprim[1]
            #print(aprim)
            x = abs(xpoint-aprim[0][0][0])
            y = abs(ypoint-aprim[0][0][1])
            coordstemp.append((x, y))
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
            #Display the resulting frame
        cv2.imshow("Framefield", framefield)
        k = cv2.waitKey(1)

        if (i % fps) == 0:
            #print("EN BILD UT")
            print("--- %s seconds ---" % (time.time() - start_time))
            small_frame = frameclock[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("Small Frame", small_frame)
            error = mse(Ref_img, small_frame)
            #print(error)

            Ref_img = small_frame

            if error < 10:
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
                xcoordtemp = []
                ycoordtemp = []
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
                val = 0
                coordstemp = []

                xcoord.extend(xcoordtemp)
                ycoord.extend(ycoordtemp)

                xcoordtemp = []
                ycoordtemp = []

                #while val <= len(totcoords):
                #    plt.plot(*zip(*totcoords))
                #    plt.show()
                #    val = val + 1

                #for cor in coordstemp:
                #    totcoords.append(coordstemp[cor])

                xdist = xdist + xdisttemp
                ydist = ydist + ydisttemp
                absolutedist = absolutedist + absolutedisttemp
                #print("X-distance traveled is: %.2f dm" % xdist)
                #print("Y-distance traveled is: %.2f dm" % ydist)
                #print("Absolute distance traveled is: %.2f dm" % absolutedist)
                absolutedisttemp = 0
                Runningtemp = 0
                Walkingtemp = 0
                Joggingtemp = 0
                xdisttemp = 0
                ydisttemp = 0

        if k%256 == 27:
            # Esc pressed
            print("Escape hit, closing...")
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
            break


ProcentRunning = 100*Running/(Running + Jogging + Walking)
ProcentJogging = 100*Jogging/(Running + Jogging + Walking)
ProcentWalking = 100*Walking/(Running + Jogging + Walking)

#print(totcoords)

heatmap, xedges, yedges = np.histogram2d(xcoord, ycoord, bins=(64, 64)) # Bins sätter gränserna för dimensionerna
# 64, 64 ger 6hx64 heatmap.
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.title("Heatmap over player movement")
plt.ylabel("Kortsida")
plt.xlabel("Långsida")
plt.imshow(heatmap, extent=extent)
plt.show()

#plt.plot(*zip(*totcoords))
#plt.suptitle("Rörelse över planen")
#plt.axis([0, 50, 0, 40])
#plt.show()

#plt.scatter(*zip(*totcoords))
#plt.show()

print("X-distance traveled is: %.2f dm" % xdist)
print("Y-distance traveled is: %.2f dm" % ydist)
print("Absolute distance traveled is: %.2f dm" % absolutedist)

print("Procent running: %.2f procent" % ProcentRunning)
print("Procent jogging: %.2f procent" % ProcentJogging)
print("Procent walking: %.2f procent" % ProcentWalking)
