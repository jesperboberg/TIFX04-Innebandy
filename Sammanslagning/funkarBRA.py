import cv2
import numpy as np
from cv2 import plot_Plot2d
import CentroidTracker
import imutils


cap = cv2.VideoCapture('videogång.mov')
#cap = cv2.VideoCapture('ppl.mp4')
#cap = cv2.VideoCapture('test2.mov')
#cap = cv2.VideoCapture(0)
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))

#forground, background
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (256,144))

#Trying to add trackbar for threshold
def nothing(x):
    pass

cv2.namedWindow("feed")
cv2.createTrackbar("Lower", "feed", 0,255,nothing)
cv2.createTrackbar("Higher", "feed", 255,255,nothing)
    
ct = CentroidTracker.CentroidTracker()

ret, frame1 = cap.read()
ret, frame2 = cap.read()
#frame1 = imutils.resize(frame1, width=640)
#frame2 = imutils.resize(frame1, width=640)
print(frame1.shape)
while cap.isOpened():
    
    #Difference between first and second frame
    diff = cv2.absdiff(frame1, frame2)
    #Convert difference to gray scale mode, easier to use
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #Smoothing the picture
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    low = cv2.getTrackbarPos("Lower", "feed")
    high = cv2.getTrackbarPos("Higher", "feed")
    
    
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    #använd delen nedan ifall vill undersöka threshold
    #_, thresh = cv2.threshold(blur, low, high, cv2.THRESH_BINARY)
    #Obs that you can change iterations for refinement
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    #fgmask = fgbg.apply(dilated)
    
    #Finding contours, obs this is a list/array
    #contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    rectList = [] #TRYING TO COMBINE CENTROID TRACKER
    for contour in contours:
        #x,y coordinate, then width and height 
        (x, y, w, h) = cv2.boundingRect(contour)
        
        #If the area of rectangle around object is less than...
        if cv2.contourArea(contour) < 900:
            continue
        #OBS, might have to change these two depending on the circumstances
        #if cv2.contourArea(contour) > 10000:
         #   continue
        
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rect = [x,y,x+w,y+h]
        rectList.append(rect)
        #print(rect)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)
        
    
    objects = ct.update(rectList)    
    
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        #if objectID<4:
        text = "ID {}".format(objectID)
        cv2.putText(frame1, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame1, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        #cv2.putText(fgmask, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv2.circle(fgmask, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    image = cv2.resize(frame1, (256,144))
    out.write(image)
    cv2.imshow("feed", frame1)
    #cv2.imshow("feed2", fgmask)
    frame1 = frame2
    #reading a new value, this way the while loop will work
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
out.release()