import pycaret.classification as clf
import pandas as pd
from sklearn.datasets import load_svmlight_file
import logging
from os import makedirs
from datetime import datetime
import matplotlib.pyplot as plt

TRAIN_FILE = "/home/gab/projetos/machine-learning/sentiment-analysis/data/word2vec/mean/test.txt"
TEST_FILE = "/home/gab/projetos/machine-learning/sentiment-analysis/data/word2vec/mean/test.txt"

def main():

    # init logger
    makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d_%H:%M:%S', 
        handlers=[
            logging.FileHandler(f"logs/classification_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.log", mode='w'), 
            logging.StreamHandler()
        ])

    logging.info(f"Loading datasets {TRAIN_FILE} and {TEST_FILE}")
    X_train, y_train = load_svmlight_file(TRAIN_FILE)
    X_test, y_test = load_svmlight_file(TEST_FILE)

    # convert to pandas dataframe
    logging.info("Converting to pandas dataframe")
    df_train = pd.DataFrame(X_train.todense())
    df_train['label'] = y_train

    logging.info(f"Train sample:\n{df_train.head()}")

    df_test = pd.DataFrame(X_test.todense())
    df_test['label'] = y_test
    logging.info(f"Test sample:\n{df_test.head()}")

    # setup experiment
    logging.info("Setting up experiment")
    exp_clf = clf.setup(
        experiment_name='sentiment-analysis',
        session_id=123,
        log_experiment=True,
        preprocess=False,
        n_jobs=-1,
        index=False,

        target='label',
        data=df_train,
        test_data=df_test,

        fold_strategy='kfold',
        fold=10,
        fold_shuffle=True,
    )

    # train logistic regression model
    logging.info("Training model")
    lr = exp_clf.create_model('lr') # logistic regression
    svm = exp_clf.create_model('svm')   # support vector machine
    lda = exp_clf.create_model('lda')   # linear discriminant analysis

    # predict on test dataset using vote strategy    
    logging.info("Predicting on test dataset")
    blended = exp_clf.blend_models([lr, svm, lda])
    exp_clf.save_model(blended, 'blended')
    blended_pred = exp_clf.predict_model(blended)

    # plot prediction scatter
    plt.scatter(blended_pred['label'], blended_pred['prediction_label'])
    plt.xlabel('Ground truth')
    plt.ylabel('Prediction')
    plt.savefig("prediction_scatter.png")



if __name__ == "__main__":
    main()