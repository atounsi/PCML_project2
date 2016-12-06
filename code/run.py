## Importing libraries
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helpers import load_data, preprocess_data
from split_data import split_data
from preprocess import preprocess
from submit_predictions import submit_predictions
from recommender import *

if __name__ == "__main__":

    ##======= Load data ======##
    print("Loading training data")
    path_dataset = "../data/data_train.csv"
    ratings = load_data(path_dataset)
    ##========================##

    ##=== Preprocess data ====##
    print("Preprocessing data")
    ratings = preprocess(ratings)
    ##========================##

    ##====Split data into training and test data sets ====##
    print("Splitting data into train and test sets")
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=1, p_test=0.1)

    ##===Train model=======##
    print("Training model")
    num_features = 1
    lambda_user = 0.1
    lambda_item = 0.7
    gamma = 0.01
    [train_rmse, test_rmse, user_features, item_features] = matrix_factorization_SGD(train,
                                                 test, num_features, lambda_user, lambda_item, gamma)
    print("RMSE on train data: {}.".format(train_rmse))
    print("RMSE on test data: {}.".format(test_rmse))


    ##===Load test data====##
    print("Loading test data")
    path_dataset = "../data/sampleSubmission.csv"
    submission_ratings = load_data(path_dataset)


    ##====Generate predictions for test data====##
    print("Generate predictions")
    nz_row, nz_col = submission_ratings.nonzero()
    nz = list(zip(nz_row, nz_col))
    prediction = [(np.dot(item_features[d,:],(user_features[:,n]))) for (d,n) in nz]
    prediction = np.round(prediction,3)


    ##==== Create submission file=====##
    print("Creating submission file")
    sampleSubmissionFilename = '../data/sampleSubmission.csv'
    outputFilename = 'submit.csv'
    submit_predictions(prediction, outputFilename, sampleSubmissionFilename)




