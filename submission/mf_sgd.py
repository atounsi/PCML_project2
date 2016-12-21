import numpy as np
from helpers import *

def matrix_factorization_SGD(train, test, num_features=1, lambda_user=0.1, lambda_item=0.1, gamma=0.01, max_iter=10):
    """matrix factorization by SGD."""
    
    num_epochs = max_iter     # number of full passes through the train set
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
    
    print("Learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        print("Running {} / {} iterations".format(it+1, num_epochs))
        for d, n in nz_train:
        # ***************************************************
            prediction = item_features[d,:].dot(user_features[:,n])
            #gradient
            gradient = np.zeros(((num_items + num_users),num_features))
            prediction_error = (train[d,n] - prediction)
            #gradient entries for W
            gradient[d,:] = -(prediction_error)*(user_features[:,n].T)
            #gradient entries for Z
            gradient[num_items+n,:] = -(prediction_error)*(item_features[d,:])
            #update
            item_features = item_features - gamma*gradient[:num_items,:]
            user_features = user_features - gamma*gradient[num_items:,:].T
            
        train_rmse = compute_error(train, user_features, item_features, nz_train)
        # ***************************************************

        print("iter: {}, RMSE on training set: {}.".format(it+1, np.round(train_rmse,5)))
        learning_curve_train.append(train_rmse)
        learning_curve_test.append(compute_error(test, user_features, item_features, nz_test))
        
        # decrease step size
        gamma /= 1.2

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(np.round(test_rmse,5)))

    prediction = item_features.dot(user_features)
    return prediction, train_rmse, test_rmse, user_features, item_features

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

def init_MF_b(train, num_features):
    """init the parameter for matrix factorization."""
    num_items, num_users = train.shape
    
    user_features = 0.1* np.ones ((num_features, num_users)) 
    item_features = 0.1*np.ones ((num_features, num_items)) 

    return user_features, item_features
def compute_error_b(data, user_features, item_features, nz, b_u, b_i, b_g):

        # ***************************************************
    real_label = np.array([data[d,n] for (d,n) in nz])
    pred_array = np.dot(item_features.T,user_features) + b_u + b_i + b_g
    prediction = np.array([pred_array[d,n] for (d,n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label,prediction))
    # ***************************************************
    return rmse

def mean_user(train,test):  
    num_items, num_users = train.shape
    mean =  np.array(train.sum(axis=0)/train.getnnz(axis=0))[0]   
    mean_u = np.ones((num_items,num_users))*mean.T
    return mean_u

def mean_item(train,test):  
    num_items, num_users = train.shape
    mean =  np.array(train.sum(axis=1).T/train.getnnz(axis=1))[0]   
    mean_i = (np.ones((num_items,num_users)).T*mean).T
    return mean_i

def mean_global(train,test):
    num_items, num_users = train.shape
    mean = (train.sum())/train.nnz
    mean_g = np.ones((num_items,num_users))*mean
    return mean_g

def matrix_factorization_SGD_b(train, test, num_features, lambda_user, lambda_item, gamma):
    """matrix factorization by SGD."""
    it = 0    # number of full passes through the train set # use test rmse
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    b_u = mean_user(train,test)
    b_i = mean_item(train,test)
    b_g = mean_global(train,test)

    errors[0] = compute_error_b(train, user_features, item_features, nz_train, b_u, b_i, b_g)
    num_items, num_users = train.shape
    error = 100.5
    error_new = 90.5
        
    print("learn the matrix factorization using SGD...")
    while ( error > error_new and it < 10):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        error = error_new
        
        # decrease step size
        gamma /= 1.2

        for d, n in nz_train:
            
            prediction = item_features[:,d].dot(user_features[:,n])
            #gradient
            prediction_error = (train[d,n] - b_u[d,n] - b_i[d,n] - b_g[d,n] - prediction )
            #gradient entries for W
            gradient_w= -(prediction_error)*(user_features[:,n].T) + lambda_item * (item_features[:,d].T)
            #gradient entries for Z
            gradient_z = -(prediction_error)*(item_features[:,d]) + lambda_user * (user_features[:,n].T)
            
            b_u[d,n] += gamma*(prediction_error - lambda_user*b_u[d,n])
            b_i[d,n] += gamma*(prediction_error - lambda_item*b_i[d,n])
            
            #update
            item_features[:,d] -=  gamma*gradient_w.T
            user_features[:,n] -=  gamma*gradient_z.T

        
        train_rmse = compute_error_b(train, user_features, item_features, nz_train, b_u, b_i, b_g)
        print("RMSE on test data: {}.".format(np.round(train_rmse,5)))
        error_new = compute_error_b(test, user_features, item_features, nz_test, b_u, b_i, b_g)

        it += 1
        errors.append(train_rmse)
    test_rmse = compute_error_b(test, user_features, item_features, nz_test, b_u, b_i, b_g)
    print("RMSE on test data: {}.".format(np.round(test_rmse,5)))

    pred = item_features.T.dot(user_features)+b_u+b_i+b_g
    return pred, train_rmse, test_rmse

