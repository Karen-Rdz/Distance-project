import numpy as np
import cv2 as cv
from networktables import NetworkTables

NetworkTables.initialize(server='10.46.35.2')
table = NetworkTables.getTable('EctoVision')
# orb = cv.ORB_create()

img1 = cv.imread('VisionObject.png') 
img1 = cv.Canny(img1,100,200)
# img1 = cv.resize(img1, (0,0), fx=0.5, fy=0.5)
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img1, contours1, hierarchy = cv.findContours(img1, cv.RETR_EXTERNAL ,cv.CHAIN_APPROX_SIMPLE )
img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
img2 = cv.drawContours(img1,contours1,0,(0,0,255),2)
corners1 = []
for c in contours1:
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect) 
    box = np.int0(box) 

    corners1.append(box[0])
    corners1.append(box[1])
    corners1.append(box[2])
    corners1.append(box[3])

# kp1, des1 = orb.detectAndCompute(img1,  None)
# kp1, des1 = orb.compute(img1,kp1)

# cv.imshow("Source", img1)
# img2= cv.drawKeypoints(img1, kp1, None, color = (0,255,0), flags = 0)
cv.imshow ("keypoints", img2)

cap = cv.VideoCapture(0)
# cap.set (10, 0)
# cap.set(12,255)
# cap.set(15, -10)

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

# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = False)

# clusters = np.array([des1])
# bf.add(clusters)

while (1):
    _, frame = cap.read ()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # frame = cv.imread('SceneVisionTarget.png',0)
    # frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)

    frame = cv.inRange (frame, lower_blue, upper_blue)
    frame = cv.Canny(frame, 100, 200)

    frame, contours, hierarchy = cv.findContours(frame, cv.RETR_EXTERNAL ,cv.CHAIN_APPROX_SIMPLE )
    corners2 = []
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for c in contours:
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect) 
        box = np.int0(box) 
        print (rect)

        corners2.append(box[0])
        corners2.append(box[1])
        corners2.append(box[2])
        corners2.append(box[3])

    cv.imwrite("LastFrame.jpg", frame)

    # kp2, des2 = orb.detectAndCompute(frame,None)

    # if des2 is not None:
    #     matches = bf.match (des1, des2)
    #     matches = sorted(matches, key = lambda x:x.distance)
    #     good = []

    #     for m in matches:
    #         if m.distance < 60:
    #             good.append(m)
        
    #     img3 = cv.drawMatches(img1, kp1, frame, kp2, matches[:20], None)
    # # img3 = cv.resize(img3, (0,0), fx=0.2, fy=0.2)
        
    #     DetectedObject = len(good) >=  0
    #     # print (len(kp2))
    #     # print (len(good))
    #     for i in range(len(good)-1, 0, -1):
    #         m = good[i]
    #         if m.trainIdx > len(kp2)-1:
    #             good.pop(i)
    #             # print (m.trainIdx)
        
    #     # print ("____________________" + str(len(good)))

    #     for i in range(len(good)-1, 0, -1):
    #         m = good[i]
    #         if m.queryIdx > len(kp1)-1:
    #             good.pop(i)
    #     #         print (m.queryIdx)

    #     # print ("----------------------" + str(len(good)))

    #     # print (DetectedObject)
    DetectedObject  = len(corners2) == 8
    if DetectedObject:
        # print ('__________________________')
        # print (np.float32([kp1[m.queryIdx].pt for m in good]))
        # print ('--------------------------------')
        # src_pts  = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        # dst_pts  = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        src_pts  = np.float32(corners1)
        dst_pts  = np.float32(corners2)           
        # print (src_pts)
        # print ("++++++++++++++++++++++++++")
        # print (corners1)


        ## find homography matrix and do perspective transform
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        center_pts = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
        corner_pts_3d = np.float32([ [-w/2,-h/2,0],[-w/2,(h-1)/2,0],[(w-1)/2,(h-1)/2,0],[(w-1)/2,-h/2,0] ])###

        if M is None:
            continue

        dst = cv.perspectiveTransform(pts,M)

        ## draw found regions
        # frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
        frame = cv.polylines(frame, [np.int32(dst)], True, (255,255,255), 1, cv.LINE_AA)

        kinect_intrinsic_param = np.array([[1.19287305e+03, 0.00000000e+00, 7.34328867e+02],[0.00000000e+00, 1.12075134e+03, 2.54940960e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        kinect_distortion_param = np.array([[ 0.27541156, -0.43489316, -0.03132834 , 0.02542095 , 0.36158149]])

        retval, rotation, translation = cv.solvePnP(corner_pts_3d, dst.reshape (-1,2), kinect_intrinsic_param, kinect_distortion_param )
        rotation = rotation*180/np.pi

        font = cv.FONT_HERSHEY_SIMPLEX
        
        cv.putText(frame, str("translation"),(10,30), font, 0.7,(0,255,0),1,cv.LINE_AA)
        cv.putText(frame,'x:'+str(translation[0]),(250,30), font, 0.7,(0,0,255),2,cv.LINE_AA)
        cv.putText(frame,'               y:'+str(translation[1]),(350,30), font, 0.7,(0,0,255),2,cv.LINE_AA)
        cv.putText(frame,'                              z:'+str(translation[2]),(450,30), font, 0.7,(0,0,255),2,cv.LINE_AA)
        
        cv.putText(frame, str("rotation"),(10,60), font, 0.7,(0,255,0),1,cv.LINE_AA)
        cv.putText(frame,'x:'+str(rotation[0]),(250,60), font, 0.7,(0,0,255),2,cv.LINE_AA)
        cv.putText(frame,'               y:'+str(rotation[1]),(350,60), font, 0.7,(0,0,255),2,cv.LINE_AA)   
        cv.putText(frame,'                              z:'+str(rotation[2]),(450,60), font, 0.7,(0,0,255),2,cv.LINE_AA) 

        table.putNumber('TranslationX', translation[0])
        table.putNumber('TranslationY', translation[1])
        table.putNumber('TranslationZ', translation[2])

        table.putNumber('RotationX', rotation[0])
        table.putNumber('RotationY', rotation[1])
        table.putNumber('RotationZ', rotation[2])
        # cv.imshow ("Matches", img3)

        
        table.putBoolean('DetectedObject', DetectedObject)
    cv.imshow ("Frame", frame)
    k = cv.waitKey(1)
    if k == 27:
        break 
