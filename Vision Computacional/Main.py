import numpy as np
import cv2 as cv
from networktables import NetworkTables
import TargetUtils

NetworkTables.initialize(server='10.46.36.2')
table = NetworkTables.getTable('EctoVision')

img1 = cv.imread('VisionTarget.png') 
img1 = cv.Canny(img1,100,200)

corners1 = TargetUtils.getCornersVisionTarget(img1)

AvailableCameras = [0,1]

cap1 = cv.VideoCapture(AvailableCameras[0])
cap2 = cv.VideoCapture(AvailableCameras[1])
cap = cap1
k = -1
CameraInUse = 0

while cap.isOpened():
    CameraInUse = table.getNumber('CameraInUse', 0) 

    if CameraInUse == 0:
        cap = cap1

    if CameraInUse == 1:
        cap = cap2

    _, frame = cap.read ()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # frame = cv.resize(frame, (0,0), fx=0.5, fy=0.5)
    # frame = cv.imread('SceneVisionTarget.png',0)
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
    frame = cv.inRange (frame, lower_blue, upper_blue)
    frame = cv.dilate (frame, None)
    frame = cv.erode (frame, None)

    cv.imwrite("LastFrame.png", frame)

    cv.imshow ("Frame", frame)

    frame = cv.Canny(frame, 500, 700)

    corners2 = TargetUtils.getCornersVisionTarget(frame)
    frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    cv.imwrite("LastFrame.jpg", frame)
    DetectedObject  = len(corners2) == 8

    if DetectedObject:

        cv.circle(frame, tuple(corners2[0]), 2, (255, 0, 0), 2)
        cv.circle(frame, tuple(corners2[1]), 2, (0, 255, 0), 2)
        cv.circle(frame, tuple(corners2[2]), 2, (0, 0, 255), 2)
        cv.circle(frame, tuple(corners2[3]), 2, (0, 255, 255), 2)
        cv.circle(frame, tuple(corners2[4]), 2, (255, 0, 0), 2)
        cv.circle(frame, tuple(corners2[5]), 2, (0, 255, 0), 2)
        cv.circle(frame, tuple(corners2[6]), 2, (0, 0, 255), 2)
        cv.circle(frame, tuple(corners2[7]), 2, (0, 255, 255), 2)
       

        src_pts  = np.float32(corners1)
        dst_pts  = np.float32(corners2)           

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        center_pts = np.float32([ [w/2,h/2] ]).reshape(-1,1,2)
        corner_pts_3d = np.float32(
            
            [ 
            [-w/2,-h/2,0],
            
            [-w/2,(h)/2,0],

            [(w-1)/2,(h)/2,0],

            [(w-1)/2,-h/2,0] 
            
            ])

        if M is None:
            continue

        dst = cv.perspectiveTransform(pts,M)

        frame = cv.polylines(frame, [np.int32(dst)], True, (255,255,255), 1, cv.LINE_AA)

        kinect_intrinsic_param = np.array([[1.13208447e+03, 0.00000000e+00, 6.27309371e+02], [0.00000000e+00, 1.09158447e+03, 3.02019844e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        kinect_distortion_param = np.array([[ 1.27148934e-01, -1.08680934e+00, -5.28545023e-03,  8.34825455e-04, 1.94722397e+00]])

        retval, rotation, translation = cv.solvePnP(corner_pts_3d, dst.reshape (-1,2), kinect_intrinsic_param, kinect_distortion_param )
        rotation = rotation*180/np.pi

        font = cv.FONT_HERSHEY_SIMPLEX
        conversionFactor = 2187.845
        
        cv.putText(frame, str("translation"),(10,30), font, 0.5,(0,255,0),1,cv.LINE_AA)
        cv.putText(frame,'x:'+str((translation[0]+w)/conversionFactor),(150,30), font, 0.5,(0,0,255),2,cv.LINE_AA)
        cv.putText(frame,'       y:'+str(translation[1]/conversionFactor),(250,30), font, 0.5,(0,0,255),2,cv.LINE_AA)
        cv.putText(frame,'              z:'+str(translation[2]/conversionFactor),(350,30), font, 0.5,(0,0,255),2,cv.LINE_AA)
        
        cv.putText(frame, str("rotation"),(10,60), font, 0.5,(0,255,0),1,cv.LINE_AA)
        cv.putText(frame,'x:'+str(rotation[0]),(150,60), font, 0.5,(0,0,255),2,cv.LINE_AA)
        cv.putText(frame,'       y:'+str(rotation[1]),(250,60), font, 0.5,(0,0,255),2,cv.LINE_AA)   
        cv.putText(frame,'               z:'+str(rotation[2]),(350,60), font, 0.5,(0,0,255),2,cv.LINE_AA) 

        table.putNumber('TranslationX', (translation[0]+w)/conversionFactor)
        table.putNumber('TranslationY', translation[1]/conversionFactor)
        table.putNumber('TranslationZ', translation[2]/conversionFactor)

        table.putNumber('RotationX', rotation[0])
        table.putNumber('RotationY', rotation[1])
        table.putNumber('RotationZ', rotation[2])
        
        table.putBoolean('DetectedObject', DetectedObject)
    
    cv.imshow ("Frame", frame)
    k = cv.waitKey(1)
    if k == 27:
        break