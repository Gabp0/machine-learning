#!/usr/bin/python
# -*- encoding: iso-8859-1 -*-

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import time


def knn_classifier(fname, k, metric):

    # loads data
    print ("Loading data...")
    data = np.loadtxt(fname)
    X_data = data[:, 1:]
    y_data = data[:,0]

    print ("Spliting data...")
    X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.3, random_state = 5)

    # cria um kNN
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)

    print ('Fitting knn')
    neigh.fit(X_train, y_train)

    # predicao do classificador
    print ('Predicting...')
    y_pred = neigh.predict(X_test)

    # mostra o resultado do classificador na base de teste
    print ('Accuracy: ',  neigh.score(X_test, y_test))

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print (cm)
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: knn.py <file>")
            
    for k in range(1, 10):
        for metric in ['cityblock', 'euclidean', 'cosine', 'l1']:
            try:
                print(f"K = {k}, Metric = {metric}")
                start_time = time()
                knn_classifier(sys.argv[1], 3, metric)
                print(f"Time ellapsed: {time() - start_time}")
            except Exception:
                pass


