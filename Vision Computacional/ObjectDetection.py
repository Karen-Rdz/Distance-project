#dfrom __future__ import division
from networktables import NetworkTables
import numpy as np
from numpy import array
import cv2

def  nothing(x):
    pass

# To use another camera pass 1
cap = cv2.VideoCapture(0)

# Initialize NetworkRables ('Name of the server')
NetworkTables.initialize(server='10.46.35.68')

# Get table from the server ('Name of the table')
table = NetworkTables.getTable('ObjectDegrees')

cv2.namedWindow ('sett')
fileLow = open("LowerRange.txt", "r+")
fileUpp = open("UpperRange.txt", "r+")

# Read the files with the range of colors
hl = fileLow.readline()
sl = fileLow.readline()
vl = fileLow.readline()
hu = fileUpp.readline()
su = fileUpp.readline()
vu = fileUpp.readline()

# Do the arrays of the range
lower_blue = np.array([int(hl), int(sl), int(vl)])
upper_blue = np.array([int(hu), int(su), int(vu)])

# cv2.createTrackbar ('nfeatures', 'sett', 11, 31, nothing)
# cv2.createTrackbar ('edgeThreshold', 'sett', 10, 31, nothing)

# Create sliders for values of the mask
# cv2.createTrackbar ('HL', 'sett', 50, 180, nothing)
# cv2.createTrackbar ('SL', 'sett', 0, 255, nothing)
# cv2.createTrackbar ('VL', 'sett', 0, 255, nothing)

# cv2.createTrackbar ('HU', 'sett', 50, 180, nothing)
# cv2.createTrackbar ('SU', 'sett', 255, 255, nothing)
# cv2.createTrackbar ('VU', 'sett', 255, 255, nothing)

while (1):
    k = cv2.waitKey(5) & 0xFF
    _, frame = cap.read()

    # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    nf = cv2.getTrackbarPos ('nfeatures', 'sett')
    et = cv2.getTrackbarPos ('edgeThreshold', 'sett')

    # Get the values for the mask
    # hl = cv2.getTrackbarPos ('HL', 'sett')
    # sl = cv2.getTrackbarPos ('SL', 'sett')
    # vl = cv2.getTrackbarPos ('VL', 'sett')

    # hu = cv2.getTrackbarPos ('HU', 'sett')
    # su = cv2.getTrackbarPos ('SU', 'sett')
    # vu = cv2.getTrackbarPos ('VU', 'sett')

    # Create arrays for the mask
    # lower_blue = np.array([hl, sl, vl])
    # upper_blue = np.array([hu, su, vu])

    # Do the mask
    mask = cv2.inRange (hsv, lower_blue, upper_blue)
    mask = cv2.dilate (mask, None)
    mask = cv2.erode (mask, None)

   # Mark the center of the video
    pixels = frame.shape
    X = pixels[0]
    Y = pixels[1]
    CY = (pixels[0]/2)
    CX = (pixels[1]/2)
    cv2.circle(frame, (int(CX), int(CY)), 7, (0, 0, 255), -1)
    mask = cv2.GaussianBlur(mask,(11, 11),0)

    # Do cotours of the image and define the center
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if area > 2500:
            x,y,w,h = cv2.boundingRect(cnt)
            cx = x+w/2
            cy = y+h/2
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame, (int(cx), int(cy)), 7, (255, 0, 0), -1)
            px = 60/pixels[1]
            py = 60.0/pixels[0]
            degreesX = (cx - CX)*px
            degreesY = (cy - CY)*py
            # y = 9.3320*10^-6x^3 - 0.0003x^2 - 1.6402x + 382.3345 '
            distancia = (9.3320*10)**(-6*(cy**3)) - 0.0003*(cy**2) - 1.6402*cy + 382.3345
            print (distancia)
            # Put a number into the Table ('Name of the number', the number)
            table.putNumber('DegreesX', degreesX)
            table.putNumber('DegreesY', degreesY)
        else:
            print ('Can not find a target')

        # rect = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(frame,[box], 0, (0, 255, 0), 2)
        # for contours in cnt:
        #     M = cv2.moments(cnt)
        #     cx = float(M["m10"] // M["m00"])
        #     cy = float(M["m01"] // M["m00"])
        #     cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        #     cv2.circle(frame, (int(cx), int(cy)), 7, (255, 0, 0), -1)

    # Show all the windows
    cv2.imshow ('frame', frame)
    cv2.imshow ('mask', mask)

    # Write on the files the range of colors
    if k == ord('s'):
        fileLow.write(str(hl))
        fileLow.write('\n')
        fileLow.write(str(sl))
        fileLow.write('\n')
        fileLow.write(str(vl))
        fileLow.write('\n')
        fileUpp.write(str(hu))
        fileUpp.write('\n')
        fileUpp.write(str(su))
        fileUpp.write('\n')
        fileUpp.write(str(vu))
    if k == 27:
        break

# Close the files and the windows
fileLow.close()
fileUpp.close()
cv2.destroyAllWindows()
