import numpy as np
from helpers import *

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""  
    # ***************************************************
    num_items, num_users = train.shape
    user_features = np.random.randint(low=0, high=5, size=(num_features, num_users))
    item_features = np.random.randint(low=0, high=5, size=(num_items, num_features))
    # ***************************************************
    return 1.0*user_features, 1.0*item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    real_label = np.array([data[d, n] for (d, n) in nz])
    prediction = np.array([(np.dot(item_features[d, :], (user_features[:, n]))) for (d, n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label, prediction))
    # ***************************************************
    return rmse

def matrix_factorization_SGD(train, test, num_features, lambda_user, lambda_item, gamma):
    """matrix factorization by SGD."""
    # define parameters
    #gamma = 0.01
    #num_features = 10   # K in the lecture notes
    #lambda_user = 0.1
    #lambda_item = 0.7
    num_epochs = 10     # number of full passes through the train set
    #errors = [0]
    learning_curve_train = []
    learning_curve_test = []
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    num_items, num_users = train.shape
    
    #print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        #gamma /= 1.2
        print("Running {} / {} epochs".format(it, num_epochs))
        for d, n in nz_train:
        # ***************************************************
            prediction = item_features[d,:].dot(user_features[:,n])
            #gradient
            gradient = np.zeros(((num_items + num_users),num_features))
            prediction_error = (train[d,n] - prediction)
            #print(prediction_error)
            #gradient entries for W
            gradient[d,:] = -(prediction_error)*(user_features[:,n].T)
            #gradient entries for Z
            gradient[num_items+n,:] = -(prediction_error)*(item_features[d,:])
            
            #update
            item_features = item_features - gamma*gradient[:num_items,:]
            user_features = user_features - gamma*gradient[num_items:,:].T
            
        train_rmse = compute_error(train, user_features, item_features, nz_train)
        # ***************************************************

        #print("iter: {}, RMSE on training set: {}.".format(it, train_rmse))
        learning_curve_train.append(train_rmse)
        learning_curve_test.append(compute_error(test, user_features, item_features, nz_test))
        
        # decrease step size
        gamma /= 1.2

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    #print("RMSE on test data: {}.".format(test_rmse))
    
    return train_rmse, test_rmse, user_features, item_features