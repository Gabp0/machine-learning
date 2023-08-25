import cv2
import numpy as np
from sys import argv
from os.path import isfile

# hog parameters
HEIGHT = 24
WIDTH = 24
CELL_SIZE = (6, 6)
BLOCK_SIZE = (CELL_SIZE[0]*2, CELL_SIZE[1]*2)
NUM_BINS = 9

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

def hog_init():
    """
    Inicializa o descritor HOG (Histrogram of Oriented Gradients) 
    """

    winSize = (WIDTH,HEIGHT)    
    cellSize = CELL_SIZE    
    blockSize = BLOCK_SIZE
    blockStride = cellSize
    nbins = NUM_BINS

    return cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, 1, -1, 0, 0.2, 1, 64, True)

def feature_extractor(image, hog):
    """
    Extrator que usa o HOG para 
    gerar o vetor de caracteristicas
    """
    
    image = cv2.resize(image, (WIDTH,HEIGHT))
    retval, thrsh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    if not retval:
        return

    rotated, _ = correct_skew(thrsh)
    descriptor = hog.compute(image)
    
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
    fnum = len(lines)
    if fnum > 0:
        print(f"Found {fnum} files.")
    else:
        print("No files found.")
        return

    i = 1
    for line in lines:
        filename, label = line.split()[0:2]
        filename = f"digits/{filename}"

        if not isfile(filename):
            continue
        
        print(f"File {i}/{fnum}...")
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        feature_vector = feature_extractor(img, hog)

        if feature_vector != None:
            output_file.write(f"{label} {feature_vector}\n")

        i += 1

    print("Done.")

if __name__ == "__main__":
    main()