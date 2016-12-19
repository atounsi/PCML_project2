## Importing libraries
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
from helpers import load_data, preprocess_data
from split_data import *
from preprocess import preprocess
from submit_predictions import submit_predictions

from linear_corrector import *
from recommender import *
import argparse
from cross_validation import *
from linear_corrector import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'method', metavar='int', type=int, help='an integer in the range 0..5')
    #parser.add_argument(
    #    'num_features', metavar='float', type=int, help='an integer , number of features')
    #parser.add_argument(
    #    'lambda_user', metavar='float', type=float, help='float : penalize users')
    #parser.add_argument(
    #    'lambda_item', metavar='float', type=float, help='float : penalize items')
    #parser.add_argument(
    #    'gamma', metavar='float', type=float, help='learning rate for SGD')
    parser.add_argument("-v", "--submit", action="store_true",
                    help="submit the results")
    args = parser.parse_args()
    method = args.method
    num_features = 1 #args.num_features
    lambda_user = 0.01 #args.lambda_user
    lambda_item = 0.05 #args.lambda_item
    gamma = 0.01 #args.gamma
    
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
    #print("Splitting data into train and test sets")
    #num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    #num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    #valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=1, p_test=0.1)

    print("Splitting data into train and test sets")
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    if method < 4 :
        valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=1, p_test=0.1)
    else: #numpy style
        train, test = split_data_numpy(ratings, p_test = 0.1, seed=12) 
        
    ##===Train model=======##
    print("Training model")
    #num_features = 10
    #lambda_user = 0.1
    #lambda_item = 0.7
    #gamma = 0.01
    if method == 0:
        ## SGD
        [train_rmse, test_rmse, user_features, item_features] = matrix_factorization_SGD(train, test, num_features, lambda_user, lambda_item, gamma) 
    elif method == 1:            
        ## ALS
        [train_rmse, test_rmse, user_features, item_features] = ALS(train, test, num_features, lambda_user, lambda_item) 
    elif method == 2:
        ## CCD    
        [train_rmse, test_rmse, user_features, item_features] = CCD(train, test, 
                                                                num_features, lambda_user, lambda_item)
    elif method == 3:
        ## CCD++    
        ##[train_rmse, test_rmse, user_features, item_features] = CCDplus(train, test, 
        ##                                                        num_features, lambda_user, lambda_item)
        K=10
        [train_rmse, test_rmse, user_features, item_features] = cross_validation_run(ratings, method, K, num_features, lambda_user, lambda_it
         pred = item_features.dot(user_features);
    elif method == 4:
        ## ALS_numpy   
        [pred, train_rmse, test_rmse] =                         ALS_numpy(train, test, 
                                                                num_features, lambda_user, lambda_item)
    elif method == 5:
        ## ALS_biased   
        [pred, train_rmse, test_rmse] =                         ALS_biased(train, test, 
                                                                num_features, lambda_user, lambda_item)
    elif method == 4:
        ## ALS_numpy   
        [pred, train_rmse, test_rmse] =                         ALS_numpy(train, test, 
                                                                num_features, lambda_user, lambda_item)
    elif method == 5:
        ## ALS_biased   
        [pred, train_rmse, test_rmse] =                         ALS_biased(train, test, 
                                                                num_features, lambda_user, lambda_item)
    else:
        print("Incorrect method, 0-SGD, 1-ALS, 2-CCD ...")
        
    '''
    need pred train test (tried with numpy style)
    
    '''
    
    pred_ready = linear_corrector(pred, train, test)
    
    
    pred_corrected =  bound_corrector(pred_ready)  
    

    print("RMSE on train data: {}.".format(train_rmse))
    print("RMSE on test data: {}.".format(test_rmse))
    

    if args.submit:
        ##===Load test data====##
        print("Loading test data")
        path_dataset = "../data/sampleSubmission.csv"
        submission_ratings = load_data(path_dataset)


        ##====Generate predictions for test data====##
        print("Generate predictions")
        prediction = sp.lil_matrix(submission_ratings.get_shape())
        nz_row, nz_col = submission_ratings.nonzero()
        nz = list(zip(nz_row, nz_col))

          
        for i in range(len(nz_row)):
            if method < 4 :
                prediction[nz_row[i], nz_col[i]] = np.dot(item_features[nz_row[i],:], user_features[:,nz_col[i]])
            else:
                prediction[nz_row[i], nz_col[i]] = pred_corrected[nz_row[i],nz_col[i]]

        ##==== Create submission file=====##
        print("Creating submission file")
        sampleSubmissionFilename = '../data/sampleSubmission.csv'
        outputFilename = 'submit.csv'
        submit_predictions(prediction, outputFilename, sampleSubmissionFilename)
    



