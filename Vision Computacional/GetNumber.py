from networktables import NetworkTables
import numpy as np 
from numpy import array
import cv2

NetworkTables.initialize(server="10.46.35.68")
table = NetworkTables.getTable("ObjectDegrees")
while (1):
    k = cv2.waitKey(5) & 0xFF

    degreesX = table.getNumber("DegreesX", 0)

    print degreesX

    if k == 27:
        break
    