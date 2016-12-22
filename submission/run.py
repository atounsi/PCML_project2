###### Importing libraries ########################
### The following Python modules are required
#### 1. numpy 2. scipy 3. argparse 4. itertools
#### No external libraries are required
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import argparse

from helpers import load_data, preprocess_data
from split_data import *
from submit_predictions import submit_predictions
from cross_validation import *

from mf_sgd import *
from mf_als import *
from mf_ccd import *
from user_collaborative_filter import *
from ensemble_als_userfilter import *
from linear_corrector import *
#####################################################


if __name__ == "__main__":

    ################################# Arguments ###################################################
    ### This script is used to generate the submit.csv predictions used for Kaggle submission
    ### This script requires 1 mandatory argument and 1 optional argument.
    ##### Command to run: python run.py method_id
    ##### The 'method_id' argument is mandatory and can take one of the following values
    ########## 0-Ensemble, 1-ALS, 2-CCD, 3-CCD++, 4-UserCollaborativeFiltering 5-SGD
    ########## 0-Ensemble produces the best predictions (Kaggle submission) in terms of RMSE
    ##### The script will generate .csv submission file by default. This can be
    ########## disabled by appending --nosubmit while running the script. python run.py 0 --nosubmit
    ################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('method', metavar='int', type=int, help='an integer in the range 0..5')
    parser.add_argument("-v", "--nosubmit", action="store_true", help="Do not generate .csv predictions")
    args = parser.parse_args()
    method = args.method
    ################################################################################################

    ################################# Model parameters ##############################################
    ### The model parameters can be set here
    num_features        = 2        # Number of latent features in matrix factorization model
    lambda_user         = 0.01      # Regularisation parameter for user feature
    lambda_item         = 0.01      # Regularisation parameter for item feature
    gamma               = 0.01     # Learning rate (only for SGD)
    num_user_neighbours = 50       # Number of users in neighbourhood (for user collaborative filtering)
    lambda_ridge        = 0.01     # Regularisation parameter for ridge regression
    weight_als          = 0.75     # Weight for producing ensemble prediction
    max_iter            = 50       # Maximum number of iterations of training 
    K                   = 10       # K-fold cross validation (valid only when cross_validation_enabled=1)
    cross_validation_enabled=0     # Enable(1)/Disable(0) cross validation
    #################################################################################################


    #############======= Load data ======##################
    ### We load the training data here
    print("**********************************************")
    print("Loading training data")
    path_dataset = "data/data_train.csv"
    ratings = load_data(path_dataset)
    #######################################################


    ###################====Split data into training and test data sets ====##################
    ### The data is split into training and test sets
    print("**********************************************")
    print("Splitting data into train and test sets")
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    valid_ratings, train, test = split_data(ratings, num_items_per_user, num_users_per_item, min_num_ratings=1, p_test=0.1)
    #########################################################################################


    ##################################################################################################################
    ##################################===Train model=======###########################################################
    ### We train the model here using one of the methods (0-Ensemble, 1-ALS, 2-CCD, 3-CCD++, 4-UserCollaborativeFiltering 5-SGD)
    print("**********************************************")
    print("Training model")
    if cross_validation_enabled == 0:   ### Cross validation disabled
        if method == 0:
            ## Ensemble (ALS_biased + user_collaborative_filter)
            train, test = split_data_numpy(ratings, p_test=0.1, seed=46)
            ## Requires arrays as input, hence using numpy version of split_data
            [pred, ensemble_rmse, user_feature, item_features] = ensemble_als_userfilter(ratings, train, test, 
                                                                    num_features, lambda_user, lambda_item, max_iter,
                                                                    lambda_ridge, weight_als, num_user_neighbours, args.nosubmit)
        elif method == 1:  
            ## ALS with bias
            [pred, train_rmse, test_rmse, user_features, item_features] = ALS_biased(train.toarray(), test.toarray(), 
                                                                            num_features, lambda_user, lambda_item, max_iter)
            pred =  bound_corrector(pred)          
            #[prediction, train_rmse, test_rmse] = ALS(train, test, num_features, lambda_user, lambda_item)
            #[pred, train_rmse, test_rmse] = ALS_numpy(train, test, num_features, lambda_user, lambda_item)
        elif method == 2:
            ## CCD    
            [train_rmse, test_rmse, user_features, item_features] = CCD(train, test, num_features, lambda_user, lambda_item)
            pred = item_features.dot(user_features)
        elif method == 3:
            ## CCD++    
            [train_rmse, test_rmse, user_features, item_features] = CCDplus(train, test, num_features, lambda_user, lambda_item)
            pred = item_features.dot(user_features)
        elif method == 4:
            ## User collaborative filtering 
            train, test = split_data_numpy(ratings, p_test=0.1, seed=46)
            ## Requires arrays as input, hence using numpy version of split_data
            pred = user_collaborative_filter(train, test, num_user_neighbours, args.nosubmit)
        elif method == 5:
            ## SGD
            [pred, train_rmse, test_rmse] = matrix_factorization_SGD(train, test, num_features, lambda_user, lambda_item, gamma, max_iter) 
            #[pred, train_rmse, test_rmse] = matrix_factorization_SGD_b(train, test, num_features, lambda_user, lambda_item, gamma) 
        else:
            print("Incorrect method, 0-Ensemble, 1-ALS, 2-CCD, 3-CCD++, 4-UserCollaborativeFiltering 5-SGD")
    else:   ## Cross validation enabled
        [train_rmse_arr, val_rmse_arr, train, test, tr_min, vr_min, us_ft_min, itm_ft_min] = cross_validation_run(ratings, 
                                                            method, K, num_features,lambda_user, lambda_item, gamma, max_iter, 
                                                            lambda_ridge, weight_als, num_user_neighbours,args.nosubmit)
        print("Train RMSE for k-fold CV: {}".format(train_rmse_arr))
        print("Test  RMSE for k-fold CV: {}".format(val_rmse_arr))
    print("**********************************************")
    ##################################################################################################################
    

    #############################################################################
    #############==== Generate predictions and submission file ==== #############
    ### We load the test indices and generate the submission file here
    if (not args.nosubmit) and (not cross_validation_enabled):
        ##===Load test data====##
        print("**********************************************")
        print("Loading test data")
        path_dataset = "data/sampleSubmission.csv"
        submission_ratings = load_data(path_dataset)

        ##==== Create submission file=====##
        print("**********************************************")
        print("Creating submission file")
        prediction = sp.lil_matrix(pred)
        sampleSubmissionFilename = 'data/sampleSubmission.csv'
        outputFilename = 'submit.csv'
        submit_predictions(prediction, outputFilename, sampleSubmissionFilename)
        print("Submission file created !!")
        print("**********************************************")
    #############################################################################

