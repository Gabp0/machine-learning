#!/usr/bin/python3
# -*- encoding: iso-8859-1 -*-

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import distance_metrics
from time import time
import pandas as pd

def knn_classifier(fname, k, metric):

    # loads data
    print ("Loading data...")
    data = np.loadtxt(fname)
    X_data = data[:, 1:]
    y_data = data[:,0]

    print ("Spliting data...")
    X_train, X_test, y_train, y_test =  train_test_split(X_data, y_data, test_size=0.5, random_state = 5)

    # cria um kNN
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric)

    print ('Fitting knn')
    neigh.fit(X_train, y_train)

    # predicao do classificador
    print ('Predicting...')
    y_pred = neigh.predict(X_test)

    # mostra o resultado do classificador na base de teste
    acc = neigh.score(X_test, y_test)
    print ('Accuracy: ',  acc)

    # cria a matriz de confusao
    cm = confusion_matrix(y_test, y_pred)
    print (cm)
    print(classification_report(y_test, y_pred))

    return acc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Use: knn.py <file>")
            
    accuracy_df = pd.DataFrame(columns=['k'] + list(distance_metrics().keys()))
    time_df = pd.DataFrame(columns=['k'] + list(distance_metrics().keys()))
    
    for k in range(1, 10):
    # for k in range(4, 5):
        acc_row = {'k': k}
        time_row = {'k': k}
        for metric in distance_metrics().keys():
        # for metric in ['cosine']:
            try:
                print(f"K = {k}, Metric = {metric}")

                start_time = time()
                acc = knn_classifier(sys.argv[1], k, metric)
                ts = time() - start_time

                print(f"Time ellapsed: {ts}")
                acc_row[metric] = acc
                time_row[metric] = ts
            except Exception:
                pass
        accuracy_df = pd.concat([accuracy_df, pd.DataFrame([acc_row])])
        time_df = pd.concat([time_df, pd.DataFrame([time_row])])

    # remove colunas com nan
    accuracy_df = accuracy_df.T.dropna().T
    time_df = time_df.T.dropna().T

    print(f"Accuracy:\n{accuracy_df.to_string()}")
    print(f"Time:\n{time_df.to_string()}")

    accuracy_df.to_csv('results_acc.csv', index=False)
    time_df.to_csv('results_time.csv', index=False)