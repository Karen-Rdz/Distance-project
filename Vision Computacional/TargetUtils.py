import cv2 
import numpy as np
import math

def compareCoor(coor1, coor2):
    if coor1[0] == coor2[0]:
        if coor1[1] == coor2[1]:
            return False 
    return True

def getCornersVisionTarget(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE )
    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    corners = []
    for c in contours:
        rect = cv2.minAreaRect(c)

        areaRect = rect[1][0]*rect[1][1]
        ratio = 0.0

        if rect[1][0] == 0 or rect[1][1] == 0:
            continue

        if rect[1][0] > rect[1][1]:
            ratio = rect[1][0]/rect[1][1]
        else:
            ratio = rect[1][1]/rect[1][0]

        if ratio > 3.7 or ratio < 1.3:
            continue
        if areaRect > 300:
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            Custombox = np.copy(box)

            menorY = box[3]
            SmenorY = box[3]

            mayoresY = []

            for b in Custombox[::1]:
                if  b[1] < menorY[1]:
                    menorY = b

            for b in Custombox[::1]:
                if b[1] < SmenorY[1] and b[1] != menorY[1]:
                    SmenorY = b
            
            if menorY[0] > SmenorY[0]:
                TopRight = menorY
                TopLeft = SmenorY  
            else:
                TopRight = SmenorY
                TopLeft = menorY

            for b in Custombox[::1]:
                if compareCoor(b, menorY) and compareCoor(b, SmenorY):
                    mayoresY.append(b)

            if mayoresY[0][0] > mayoresY[1][0]:
                BottomRight = mayoresY[0]
                BottomLeft = mayoresY[1]
            else:
                BottomRight = mayoresY[1]
                BottomLeft = mayoresY[0]

            # cv2.circle(img, tuple(BottomRight), 2, (255, 0, 0), 2)

            # distance = ((rect[1][0]/2)**2 + (rect[1][1]/2)**2)**0.5
            # angle = math.atan2((rect[1][1]/2), (rect[1][0]/2))
            # angle = math.degrees(angle)

            # if abs(rect[2]) > 45 :
            #     angle = math.atan2((rect[1][0]/2), (rect[1][1]/2))
            #     angle = math.degrees(angle)
            #     angle = (angle - rect[2]) - 90
            # else:
            #     angle = math.atan2((rect[1][1]/2), (rect[1][0]/2))
            #     angle = math.degrees(angle)
            #     angle = (angle - rect[2])
                
            # offSetX = int(distance * math.cos((math.radians(angle))))
            # offSetY = int(distance * math.sin((math.radians(angle))))
        
            # # TopRight = (int(rect[0][0] + offSetX), int( rect[0][1] - offSetY))
            # # BottomLeft = (int(rect[0][0] - offSetX), int( rect[0][1] + offSetY))

            # if abs(rect[2]) > 45:
            #     angle += rect[2]*2 - 180
            # else:
            #     angle += rect[2]*2

            # offSetX = int(distance * math.cos((math.radians(angle))))
            # offSetY = int(distance * math.sin((math.radians(angle))))


            # # TopLeft = (int(rect[0][0] - offSetX), int( rect[0][1] - offSetY))
            # # BottomRight = (int(rect[0][0] + offSetX), int( rect[0][1] + offSetY))

            corners.append(TopRight)
            corners.append(TopLeft)
            corners.append(BottomRight)
            corners.append(BottomLeft)

    return corners