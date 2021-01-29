import numpy as np
import cv2 
import array
import math

img = cv2.imread("VisionTarget.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 100, 200)

img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )

gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

corners = []

# gray = cv2.drawContours(gray, contours, -1, (0,255,0), 3)
for c in contours:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect) 
    box = np.int0(box)     
    print(rect)

    TopLeftX = int(rect[0][0] - rect[1][0]/2)
    TopLeftY = int(rect[0][1] - rect[1][1]/2)

    TopRightX = int(rect[0][0] + rect[1][0]/2)
    TopRightY = int(rect[0][1] - rect[1][1]/2)


    BottomLeftX = int(rect[0][0] - rect[1][0]/2)
    BottomLeftY = int(rect[0][1] + rect[1][1]/2)


    BottomRightX = int(rect[0][0] + rect[1][0]/2)
    BottomRightY = int(rect[0][1] + rect[1][1]/2)


    distance = ((rect[1][0]/2)**2 + (rect[1][1]/2)**2)**0.5
    angle = math.atan2((rect[1][1]/2), (rect[1][0]/2))
    angle = math.degrees(angle)

    if abs(rect[2]) > 45 :
        angle = math.atan2((rect[1][0]/2), (rect[1][1]/2))
        angle = math.degrees(angle)
        angle = (angle - rect[2]) - 90
    else:
        angle = math.atan2((rect[1][1]/2), (rect[1][0]/2))
        angle = math.degrees(angle)
        angle = (angle - rect[2])

    print (angle) 
    print (math.radians(angle)) 
    offSetX = int(distance * math.cos((math.radians(angle))))
    offSetY = int(distance * math.sin((math.radians(angle))))
    print ((offSetX, offSetY))  

    TopRight = (int(rect[0][0] + offSetX), int( rect[0][1] - offSetY))
    BottomLeft = (int(rect[0][0] - offSetX), int( rect[0][1] + offSetY))
    offSetX = int(distance * math.cos((math.radians(180 - angle))))
    offSetY = int(distance * math.sin((math.radians(180 - angle))))

    TopLeft = (int(rect[0][0] - offSetX), int( rect[0][1] + offSetY))
    BottomRight = (int(rect[0][0] + offSetX), int( rect[0][1] - offSetY))

    cv2.circle(gray, TopLeft, 2, (0, 0, 255), 2 ) 
    cv2.circle(gray, TopRight, 2, (0, 255, 0), 2 ) 
    cv2.circle(gray, BottomLeft, 2, (255, 0, 0), 2 ) 
    cv2.circle(gray, BottomRight, 2, (0, 255, 255), 2 ) 

    corners.append(box[0])
    corners.append(box[1])
    corners.append(box[2])
    corners.append(box[3])

    #cv2.drawContours(gray,[box],0,(255,255,255),2)

# cv2.circle(gray, tuple(corners[0]), 2, (0, 0, 255), 2)
# cv2.circle(gray, tuple(corners[1]), 2, (0, 255, 0), 2)
# cv2.circle(gray, tuple(corners[2]), 2, (255, 0, 0), 2)
# cv2.circle(gray, tuple(corners[3]), 2, (255, 255, 0), 2) 
# cv2.circle(gray, tuple(corners[4]), 2, (100, 0, 255), 2) 
# cv2.circle(gray, tuple(corners[5]), 2, (255, 100, 0), 2) 
# cv2.circle(gray, tuple(corners[6]), 2, (255, 0, 100), 2) 
# cv2.circle(gray, tuple(corners[7]), 2, (0, 100, 0), 2) 

cv2.imshow ("Contours", gray)

cv2.waitKey()
cv2.destroyAllWindows()
