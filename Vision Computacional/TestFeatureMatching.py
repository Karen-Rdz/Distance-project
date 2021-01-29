import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv.imread('Flower2.jpg',0)          # queryImage
# Initiate SIFT detector
sift = cv.ORB_create()
# find the keypoints and descriptors with SIFT

kp1 = sift.detect(img1,  None)
kp1, des1 = sift.compute(img1,kp1)

cv.drawKeypoints(img1, kp1, img1, color = (0,255,0), flags = 2)

cv.imshow ("keypoints", img1)
cap = cv.VideoCapture(0)

while (1): 
    img2 = cv.imread('Scene3.jpg',0)
    # _, img2 = cap.read()
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    # flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.match(des1,des2)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    #-- Draw matches
    # img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    # cv.drawMatches(img1, kp1, img2, kp2, good, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Show detected matches
    # img_matches = cv.resize(img_matches, (0,0), fx=0.4, fy=0.4)
    # cv.imshow('Good Matches', img_matches)

    if len(matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        print (M)
        if M is not None:
            dst = cv.perspectiveTransform(pts,M)
            img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    img3 = cv.resize(img3, (0,0), fx=0.4, fy=0.4)
    cv.imshow("gray",img3)
    k = cv.waitKey(1)
    if k == 27:
        break 
