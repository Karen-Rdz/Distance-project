
import numpy as np 
import cv2

black = (512,512,3)
blue = (255,0,0)
green = (0,255,0)
red = (0,0,255)
purple = (150, 0,150)
yellow = (0,255,255)
white = (255,255,255)

k = cv2.waitKey(0)

font = cv2.FONT_HERSHEY_SIMPLEX

pts = np.array([[10,256], [10,450], [500,450], [500,256]])
pts = pts.reshape((-1,1,2))

#     Crea el espacio en el cual vas a trabajar
img = np.zeros(black, np.uint8)

#     cv2.line(object, CoordenadasInicio, CoordenadasFinal, color, grosor)
img = cv2.line(img, (0,0), (511, 511), blue, 30)

#     CV2.polylines (object, [arreglo], True=ConectarPuntos False=NoConetarPuntos, color)
img = cv2.polylines (img, [pts], True, yellow)

#     cv2.rectangle(object, (InicioX, FinalY), (FinalX, InicioY), color, grosor)
img = cv2.rectangle(img, (100, 400), (400, 100), green, -1)

#     cv2.circle(object, CoordenadasCentro, radio, color, grosor)
img = cv2.circle(img, (255, 255), 63, red, -1)

#     cv2.ellipse(object,CoordenadasCentro, CoordenadasCentro,(altoX, altoY), (AnguloBase, AnguloEmpieza, AnguloTermina ), color, grosor)
img = cv2.ellipse(img,(256,100), (100,80),180,0,180,purple,-1)


# cv2.putText (object, 'texto', CoordenadasInicio, TipoDeLetra, TamanoDeLetra, color, ----, ----)
cv2.putText(img, 'OpenCV', (101,450), font, 3, white, 2, cv2.LINE_AA)

cv2.imshow('Formas', img)
cv2.waitKey(0)

if k == ord('q'):
    cv2.destroyAllWindows ()
    
