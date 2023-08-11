import cv2
import numpy as np
from sys import argv
from os.path import isfile

def correct_skew(img, delta=1, limit=5):
    """
    Corrige a inclinação da imagem
    Modificado de https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr
    """

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    best_score = 0.0
    corrected = None
    best_angle = 0.0
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        data = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)

        if score > best_score:
            best_score = score
            corrected = data
            best_angle = angle

    return corrected, best_angle

def hog_init(X=24, Y=24, cellhw=6):
    """
    Inicializa o descritor HOG (Histrogram of Oriented Gradients) 
    """

    winSize = (X,Y)    
    cellSize = (cellhw,cellhw)    
    blockSize = (cellSize[0]*2, cellSize[1]*2)
    blockStride = cellSize
    nbins = 9
    signedGradients = True

    return cv2.HOGDescriptor(winSize,blockSize,blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64, signedGradients)

def feature_extractor(image, hog, size=24):
    """
    Extrator que usa o HOG para 
    gerar o vetor de caracteristicas
    """
    
    image = cv2.resize(image, (size,size))
    retval, thrsh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if not retval:
        return

    rotated, _ = correct_skew(thrsh)
    descriptor = hog.compute(rotated)
    
    feature_vector = ""
    for d in descriptor:
        feature_vector += str(d) + " "
    
    return feature_vector

def main():
    if len(argv) != 3:
        print(f"Uso {argv[0]} <foldername> <output file>")

    foldername = argv[1]
    fout = argv[2]
    
    output_file = open(fout, "w")
    img_files = open(f"{foldername}/files.txt", "r")

    hog = hog_init()

    lines = img_files.readlines()
    for line in lines:
        filename, label = line.split()[0:2]
        filename = f"digits/{filename}"

        if not isfile(filename):
            continue

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        feature_vector = feature_extractor(img, hog)
    
        output_file.write(f"{feature_vector}{label}\n")

if __name__ == "__main__":
    main()