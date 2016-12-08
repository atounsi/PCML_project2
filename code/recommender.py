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


def init_MF_ALS(train, num_features):
    """init the parameter for matrix factorization."""
    
    # ***************************************************
    num_items, num_users = train.shape
    user_mean = np.array(train.sum(axis=0)/train.getnnz(axis=0))
    user_features = np.r_[user_mean,np.random.randn(num_features-1,num_users)]
    item_mean = np.array(train.sum(axis=1).T/train.getnnz(axis=1)).T
    item_features = np.c_[item_mean,np.random.randn(num_items,num_features-1)]
    #item_features = np.zeros((num_items,num_features))
    # ***************************************************
    return 1.0*user_features,1.0*item_features

def update_user_CCD(
        residual, user_features, item_features, user, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature row."""
    # ***************************************************
    num_items,num_users = residual.shape
    num_features = item_features.shape[1]
    nnz_items = nnz_items_per_user[user]
    nz_itemindices = nz_user_itemindices[user]
    for t in np.arange(num_features):
        nom = residual[nz_itemindices,user]+ user_features[t,user]*np.c_[item_features[nz_itemindices,t]]
        nom = (item_features[nz_itemindices,t].T).dot(nom)[0,0]
        denom = lambda_user + (item_features[nz_itemindices,t].T).dot(item_features[nz_itemindices,t])
        new = nom/denom
        new_res = residual[nz_itemindices,user] - (new-user_features[t,user])*np.c_[item_features[nz_itemindices,t]]
        residual[nz_itemindices,user] = np.squeeze(np.asarray(new_res))
        user_features[t,user] = new
    # ***************************************************

def update_item_CCD(
        residual, user_features, item_features, item, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature column."""
    # ***************************************************
    num_items,num_users = residual.shape
    num_features = user_features.shape[0]
    nnz_users = nnz_users_per_item[item]
    nz_userindices = nz_item_userindices[item]
    for t in np.arange(num_features):
        nom = residual[item,nz_userindices] + item_features[item,t]*np.r_[user_features[t,nz_userindices]]
        nom = nom.dot((user_features[t,nz_userindices]).T)[0,0]
        denom = lambda_item + (user_features[t,nz_userindices]).dot(user_features[t,nz_userindices].T)
        new = nom/denom
        new_res = residual[item,nz_userindices] - (new-item_features[item,t])*user_features[t,nz_userindices]
        residual[item,nz_userindices] = np.squeeze(np.asarray(new_res))
        item_features[item,t] = new
    # ***************************************************

def compute_error_residual(residual, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    t = np.array([residual[d,n] for (d,n) in nz])
    rmse = np.sqrt(t.dot(t.T)/len(nz))
    # ***************************************************
    return rmse

# Cyclic coordinate descent
def CCD(train, test, num_features=10, lambda_user=0.1, lambda_item=0.7):
    """Cyclic coordinate descent (CCD) algorithm."""
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init CCD
    user_features, item_features = init_MF_ALS(train, num_features)
    
    # ***************************************************
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    _,nz_user_itemindices = map(list,zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _,nz_item_userindices = map(list,zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]
    max_it = 20
    
    print("learn the matrix factorization using CCD...")
    
    num_items,num_users = train.shape
    residual = train - item_features.dot(user_features)

    for it in np.arange(max_it):
        for user in np.arange(num_users):
            update_user_CCD(residual, user_features, item_features, user, lambda_user, nnz_items_per_user, nz_user_itemindices)
            
        for item in np.arange(num_features):
            update_item_CCD(residual, user_features, item_features, item, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        train_rmse = compute_error_residual(residual, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, train_rmse))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(test_rmse))
    
    return train_rmse, test_rmse, user_features, item_features