import numpy as np
import cv2

img = cv2.imread('paisaje1.jpg', 1)
cv2.imshow('imagen', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows ()
elif k == ord('s'):
    cv2.imwrite('paisaje.png', img)
    cv2.destroyAllWindows ()


