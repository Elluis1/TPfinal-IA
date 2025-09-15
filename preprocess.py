import cv2
import numpy as np

def image_to_vector(path, size=(32,32)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)  # reducir tamaño
    _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)  # Todo a 0 y 1 tomado en base a si es blanco o negro
    return img_bin.flatten()  # Transforma todo a un vector con los elementos de la linea anterior
