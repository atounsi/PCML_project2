from mf_als import *
from user_collaborative_filter import *
from split_data import *
from linear_corrector import *

def ensemble_als_userfilter(ratings, train, test, num_features = 2, lambda_user = 0.01, lambda_item = 0.01, 
                             max_als_iter=50, lambda_ridge=0.01, weight_als= 0.75, num_user_neighbours=50, nosubmit=0, seed=552):
    '''
    Function to create Ensemble model consisting of ALS and collaborative filtering
    '''
    print("=========================================")
    print("Running ensemble of ALS and User collaborative filtering...")
    #train = train.toarray()
    #test = test.toarray()
    ## Matrix factorization model using ALS
    [prediction_als, train_rmse, test_rmse, user_feature, item_features] = ALS_biased(train, test, num_features, lambda_user, 
                                                          								lambda_item, max_als_iter)
    print("=========================================")
    print("Running ridge regression...")
    ### Building features for regression
    y, y_test, tX, tX_test = feature_adding(train, test, prediction_als)
    w = ridge_regression(y, tX, lambda_ridge)
    tX_all = feature_adding_all(train, test, prediction_als)
    w_nth = np.array([1,0,0,0,0,0,0])
    pred_all = tX_all.dot(w)
    pred_als_ridge = pred_all.reshape((prediction_als.shape[0], prediction_als.shape[1]))
    #pred_als_ridge = linear_corrector(prediction_als, train, test, 5)
    
    ### Compute error with ALS + ridge regression model
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    als_ridge_rmse = compute_error_prediction(test, pred_als_ridge, nz_test)
    print("ALS_Ridge: RMSE on test data: {}.".format(np.round(als_ridge_rmse,5)))
    print("=========================================")
    
    ### User collaborative filtering
    #train, test = split_data_numpy(ratings, p_test=0.1, seed=46)
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    prediction_userCollabFilt = user_collaborative_filter(train, test, num_user_neighbours, nosubmit)
    userfilt_rmse = compute_error_prediction(test, prediction_userCollabFilt, nz_test)
    if nosubmit:
        print("User filter: RMSE on test data: {}.".format(np.round(userfilt_rmse,5)))
    print("=========================================")

    ### Creating weighted prediction of ALS and user collaborative filtering
    print("Weighted average of ALS and Userfilter...")
    weight_collab_filt = 1-weight_als
    prediction_ensemble = weight_collab_filt*prediction_userCollabFilt + weight_als*pred_als_ridge
    prediction_ensemble = bound_corrector(prediction_ensemble)
    ensemble_rmse = compute_error_prediction(test, prediction_ensemble, nz_test)
    if nosubmit:
        print("ALS + Ridge + Userfilter: RMSE on test data: {}.".format(np.round(ensemble_rmse,5)))
    print("=========================================")

    return prediction_ensemble, ensemble_rmse, user_feature, item_features