import cv2
import numpy as np
import imutils

#FUNCIÓN PARA ORDENAR PUNTOS
def Puntos_Ordenados(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    orden_y = sorted(n_puntos, key = lambda n_puntos: n_puntos[1])
    orden_x1 = orden_y[0:2]
    orden_x1 = sorted(orden_x1, key = lambda orden_x1: orden_x1[0])
    orden_x2 = orden_y[2:4]
    orden_x2 = sorted(orden_x2, key = lambda orden_x2: orden_x2[0])
    #print(orden_x1[0], orden_x1[1], orden_x2[0], orden_x2[1])
    return [orden_x1[0], orden_x1[1], orden_x2[0], orden_x2[1]]

#FUNCIÓN PARA REGIÓN DE INTERÉS
def ROI(Imagen, ancho, alto):
    Imagen_salida = None
    Gris = cv2.cvtColor(Imagen, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(Gris, 150, 255, cv2.THRESH_BINARY)
    cv2. imshow("Imagen Umbralizada", th)
    contornos = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[0:1]

    for i in contornos:
        epsilon = 0.01 * cv2.arcLength(i, True)
        aproximacion = cv2.approxPolyDP(i, epsilon, True)

        if len(aproximacion) == 4 :
            puntos = Puntos_Ordenados(aproximacion)
            puntos1 = np.float32(puntos)
            puntos2 = np.float32([[0,0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(puntos1, puntos2)
            Imagen_salida = cv2.warpPerspective(Imagen, M, (ancho, alto))
    
    return Imagen_salida

#CÓGIDO GENERAL
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)
captura.set(3, 1920) #Ancho de los fotogramas
captura.set(4, 1080) #Alto de los fotogramas
Fuente = cv2.FONT_HERSHEY_SIMPLEX

while True:
    Deteccion, Camara = captura.read()
    Camara = imutils.resize(Camara, width=480)
    Camara = cv2.flip(Camara,1)
    if Deteccion == False:
        break
    cv2.imshow("Imagen Completa", Camara)
    Hoja_Segmentada = ROI(Camara, ancho=480, alto=640)
    if Hoja_Segmentada is None:
        Hoja_Segmentada = ROI(Camara, ancho=480, alto=640)
    if Hoja_Segmentada is not None:
        puntos = []
        #Se aplica Escalado de grises
        Video_Gris = cv2.cvtColor(Hoja_Segmentada, cv2.COLOR_BGR2GRAY)
        #Se aplica Desenfoque
        Desenfoque = cv2.GaussianBlur(Video_Gris, (5,5), 1)
        #Se aplica Umbralizacion
        _, Umbral_Monedas = cv2.threshold(Desenfoque, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cv2.imshow("Umbralizacion Monedas", Umbral_Monedas)
        #Se detectan los Contornos
        contornos_monedas = cv2.findContours(Umbral_Monedas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #Se dibujan los Contornos
        cv2.drawContours(Hoja_Segmentada, contornos_monedas, -1, (0, 0, 255), 2)
        Contador_50 = 0
        Contador_100 = 0
        Contador_200 = 0
        Contador_500 = 0
        Contador_1000 = 0
        for i in contornos_monedas:
            Areas = cv2.contourArea(i)
            Momentos = cv2.moments(i)
            
            if Momentos["m00"] == 0:
                Momentos["m00"] = 1.0
            
            x = int(Momentos["m10"]/Momentos["m00"])
            y = int(Momentos["m01"]/Momentos["m00"])
            
            if Areas < 3500 and Areas > 3300:
                Contador_1000 = Contador_1000 + 1
                cv2.putText(Hoja_Segmentada, "1000$", (x, y), Fuente, 0.5, (255, 0, 0), 2)
            if Areas < 2700 and Areas > 2450:
                Contador_500 = Contador_500 + 1
                cv2.putText(Hoja_Segmentada, "500$", (x, y), Fuente, 0.5, (255, 0, 0), 2)
            if Areas < 2400 and Areas > 2200:
                Contador_200 = Contador_200 + 1
                cv2.putText(Hoja_Segmentada, "200$", (x, y), Fuente, 0.5, (255, 0, 0), 2)
            if Areas < 1950 and Areas > 1700:
                Contador_100 = Contador_100 + 1
                cv2.putText(Hoja_Segmentada, "100$", (x, y), Fuente, 0.5, (255, 0, 0), 2)
            if Areas < 1350 and Areas > 1100:
                Contador_50 = Contador_50 + 1
                cv2.putText(Hoja_Segmentada, "50$", (x, y), Fuente, 0.5, (255, 0, 0), 2)
        Total = Contador_1000*1000 + Contador_500*500 + Contador_200*200 + Contador_100*100 + Contador_50*50
        cv2.putText(Hoja_Segmentada, "Total=" + str(Total) + "$", (15, 20), Fuente, 0.75, (255, 0, 0), 2)
        cv2.imshow("Imagen Recortada", Hoja_Segmentada)
    Comando = cv2.waitKey(1)
    if Comando == ord("q"):
        break
    if Comando == ord("w"):
        contador = 1
        for i in range(len(contornos_monedas)):
            area = cv2.contourArea(contornos_monedas[i])
            print("La moneda #", str(contador), " es: ", str(area))
            contador = contador + 1

captura.release() #Cierra el dispositivo de captura
cv2.destroyAllWindows() #Cierra las ventanas