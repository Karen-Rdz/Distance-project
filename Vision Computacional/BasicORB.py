import numpy as np 
import cv2
while(1):
    
img = cv2.imread('chess.jpg')

orb = cv2.ORB_create(500, 1.2, 8, 10)

kp = orb.detect(img, None)

kp, des = orb.compute(img, kp)
img2 = cv2.imread('chess.jpg')
img2 = cv2.drawKeypoints(img, kp, img2, color= (0, 255, 0), flags=0)
cv2.imshow ('img', img2)
    k = cv2.waitKey(5)
    if k == 27:
        break