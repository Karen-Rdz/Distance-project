from __future__ import division
from networktables import NetworkTables
import numpy as np 
from numpy import array
import cv2

cap = cv2.VideoCapture(0)
fileLow = open("LowerRange.txt", "r+")
fileUpp = open("UpperRange.txt", "r+")

hl = fileLow.readline()
sl = fileLow.readline()
vl = fileLow.readline()
hu = fileUpp.readline()
su = fileUpp.readline()
vu = fileUpp.readline()

lower_blue = np.array([int(hl), int(sl), int(vl)])
upper_blue = np.array([int(hu), int(su), int(vu)])

while (1):
    k = cv2.waitKey(5) & 0xFF
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = cv2.Canny (hsv, lower_blue, upper_blue)
    mask = cv2.dilate (mask, None)
    mask = cv2.erode (mask, None)

    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,0,255), 3)

    cv2.imshow ('frame', frame)
    
    if k == 27: 
        break   

fileLow.close()
fileUpp.close() 
cv2.destroyAllWindows()