import numpy as np 
import cv2

def nothing (x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow ('frame')

# HL maximo valor es 180
# SL y VL siempre empiezan en 0
cv2.createTrackbar ('HL', 'frame', 50, 180, nothing)
cv2.createTrackbar ('SL', 'frame', 0, 255, nothing)
cv2.createTrackbar ('VL', 'frame', 0, 255, nothing)

# HU maximo valor es 180
# SU Y VU siempre empiezan en 255
cv2.createTrackbar ('HU', 'frame', 50, 180, nothing)
cv2.createTrackbar ('SU', 'frame', 255, 255, nothing)
cv2.createTrackbar ('VU', 'frame', 255, 255, nothing)

while (1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hl = cv2.getTrackbarPos ('HL', 'frame')
    sl = cv2.getTrackbarPos ('SL', 'frame')
    vl = cv2.getTrackbarPos ('VL', 'frame')

    hu = cv2.getTrackbarPos ('HU', 'frame')
    su = cv2.getTrackbarPos ('SU', 'frame')
    vu = cv2.getTrackbarPos ('VU', 'frame')
    
    lower_blue = np.array([hl, sl, vl])
    upper_blue = np.array([hu, su, vu])

    mask = cv2.inRange (hsv, lower_blue, upper_blue)
    

    cv2.imshow ('frame', frame)
    cv2.imshow ('mask', mask)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()