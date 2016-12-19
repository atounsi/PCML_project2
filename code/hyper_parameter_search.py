import numpy as np
from cross_validation import *

##TODO finish to adapt function depending on method (from notebook)

## !!! Takes long time to run
def best_lambda_user(lambda_user_arr = [0.001,0.005, 0.01, 0.05, 0.1]):
    
    K = 8        ## K-fold cross validation
    num_features = 5   # K in the lecture notes
    
    lambda_item = 0.01

    train_rmse_mean = np.zeros(len(lambda_user_arr))
    train_rmse_std = np.zeros(len(lambda_user_arr))
    validation_rmse_mean = np.zeros(len(lambda_user_arr))
    validation_rmse_std = np.zeros(len(lambda_user_arr))

    for i, lambda_user in enumerate(lambda_user_arr):
        train_rmse_arr = []
        validation_rmse_arr = []

        print('Running lambda_user={n}'.format(n=lambda_user))
        train_rmse_arr, validation_rmse_arr = cross_validation_minimalist(ratings, K, num_features, lambda_user, lambda_item)

        train_rmse_mean[i] = np.mean(train_rmse_arr)
        train_rmse_std[i] = np.std(train_rmse_arr)
        validation_rmse_mean[i] = np.mean(validation_rmse_arr)
        validation_rmse_std[i] = np.std(validation_rmse_std)

    ## Plotting results
    plt.fill_between(lambda_user_arr, train_rmse_mean - train_rmse_std,
                         train_rmse_mean + train_rmse_std, alpha=0.1, color="r")
    plt.fill_between(lambda_user_arr, validation_rmse_mean - validation_rmse_std,
                         validation_rmse_mean + validation_rmse_std, alpha=0.1, color="g")
    plt.plot(lambda_user_arr, train_rmse_mean, 'o-', color="r")
    plt.plot(lambda_user_arr, validation_rmse_mean, 'o-', color="g")
    plt.legend(('Train', 'Validation'))
    plt.xlabel('Lambda user'); plt.ylabel('RMSE');
    plt.show()
    return list(zip(lambda_user_arr, train_rmse_mean, validation_rmse_mean))  #add normal stuff
print("function 'compiled'")


def best_lambda_item(lambda_item_arr = [0.001,0.005, 0.01, 0.05, 0.1]):
    
    K = 8        ## K-fold cross validation
    num_features = 5   # K in the lecture notes
    
    lambda_user = 0.01

    train_rmse_mean = np.zeros(len(lambda_item_arr))
    train_rmse_std = np.zeros(len(lambda_item_arr))
    validation_rmse_mean = np.zeros(len(lambda_item_arr))
    validation_rmse_std = np.zeros(len(lambda_item_arr))

    for i, lambda_item in enumerate(lambda_item_arr):
        train_rmse_arr = []
        validation_rmse_arr = []

        print('Running lambda_item={n}'.format(n=lambda_item))
        train_rmse_arr, validation_rmse_arr = cross_validation_minimalist(ratings, K, num_features, lambda_user, lambda_item)

        train_rmse_mean[i] = np.mean(train_rmse_arr)
        train_rmse_std[i] = np.std(train_rmse_arr)
        validation_rmse_mean[i] = np.mean(validation_rmse_arr)
        validation_rmse_std[i] = np.std(validation_rmse_std)

    ## Plotting results
    plt.fill_between(lambda_item_arr, train_rmse_mean - train_rmse_std,
                         train_rmse_mean + train_rmse_std, alpha=0.1, color="r")
    plt.fill_between(lambda_item_arr, validation_rmse_mean - validation_rmse_std,
                         validation_rmse_mean + validation_rmse_std, alpha=0.1, color="g")
    plt.plot(lambda_item_arr, train_rmse_mean, 'o-', color="r")
    plt.plot(lambda_item_arr, validation_rmse_mean, 'o-', color="g")
    plt.legend(('Train', 'Validation'))
    plt.xlabel('Lambda item'); plt.ylabel('RMSE');
    plt.show()
    return list(zip(lambda_item_arr,train_rmse_mean,  validation_rmse_mean)) 


## !!! Takes long time to run
def best_num_features(num_features_arr = [1, 2, 4, 7, 10, 14]):
    
    K = 8        ## K-fold cross validation
    
    lambda_user = 0.01
    lambda_item = 0.01

    train_rmse_mean = np.zeros(len(num_features_arr))
    train_rmse_std = np.zeros(len(num_features_arr))
    validation_rmse_mean = np.zeros(len(num_features_arr))
    validation_rmse_std = np.zeros(len(num_features_arr))

    for i, num_features in enumerate(num_features_arr):
        train_rmse_arr = []
        validation_rmse_arr = []

        print('Running num_features={n}'.format(n=num_features))
        train_rmse_arr, validation_rmse_arr = cross_validation_minimalist(ratings, K, num_features, lambda_user, lambda_item)

        train_rmse_mean[i] = np.mean(train_rmse_arr)
        train_rmse_std[i] = np.std(train_rmse_arr)
        validation_rmse_mean[i] = np.mean(validation_rmse_arr)
        validation_rmse_std[i] = np.std(validation_rmse_std)

    ## Plotting results
    plt.fill_between(num_features_arr, train_rmse_mean - train_rmse_std,
                         train_rmse_mean + train_rmse_std, alpha=0.1, color="r")
    plt.fill_between(num_features_arr, validation_rmse_mean - validation_rmse_std,
                         validation_rmse_mean + validation_rmse_std, alpha=0.1, color="g")
    plt.plot(num_features_arr, train_rmse_mean, 'o-', color="r")
    plt.plot(num_features_arr, validation_rmse_mean, 'o-', color="g")
    plt.legend(('Train', 'Validation'))
    plt.xlabel('num_features'); plt.ylabel('RMSE');
    plt.show()
    return list(zip(num_features_arr, train_rmse_mean, validation_rmse_mean)) 
print("function 'compiled'")