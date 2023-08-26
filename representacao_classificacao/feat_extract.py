# Gabriel de Oliveira Pontarolo, GRR20203895
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
        print(f"File {i}/{fnum}, loading {filename} ...")
        
        if not isfile(filename):
            print(f"File {filename} not found.")
            continue
        
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        feature_vector = feature_extractor(img, hog)

        if feature_vector != None:
            output_file.write(f"{label} {feature_vector}\n")

        i += 1

    print("Done.")

if __name__ == "__main__":
    main()