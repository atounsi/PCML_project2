import numpy as np
from helpers import *
import itertools


def init_MF_ALS(train, num_features, factor_features=0.1):
    num_items, num_users = train.shape
    user_features = factor_features*np.ones((num_users,num_features))
    item_features = factor_features*np.ones((num_items,num_features))
    return user_features, item_features

def init_MF_ALS_biased(train, num_features, factor_features=0.1, factor_biases=1):
    num_items, num_users = train.shape
    user_features = factor_features*np.ones((num_users,num_features))
    item_features = factor_features*np.ones((num_items,num_features))
    user_biases = factor_biases*np.ones(num_users)
    item_biases = factor_biases*np.ones(num_items)
    return user_features, item_features, user_biases, item_biases

def data_user_biased(data, user_biases):
    data_user_biased = data - user_biases
    return data_user_biased

def data_item_biased(data, item_biases):
    data_item_biased = (data.T - item_biases).T
    return  data_item_biased

def update_item_biased_feature(train, user_features, user_biases, lambda_item, nnz_users_per_item, nz_item_userindices):
    
    num_users, num_features = user_features.shape
    num_items = train.shape[0]
    ones_biases = np.array([np.ones(num_users)])
    item_biases = np.zeros(num_items)
    item_features = np.zeros((num_items,num_features))         
        
    for item in np.arange(num_items): 
        nnz_users = nnz_users_per_item[item]
        nz_userindices = nz_item_userindices[item]
        nz_userfeatures = user_features[nz_userindices,:]
        nz_onesbiases = ones_biases[:,nz_userindices]
        nz_userbiases = user_biases[nz_userindices]
    
    
        Xt = np.concatenate((nz_onesbiases, nz_userfeatures.T), axis=0)
        A = Xt.dot(Xt.T) + lambda_item*nnz_users*np.eye(num_features+1)  
        train_item = (train[item,nz_userindices])
        b = Xt.dot(data_user_biased(train_item, nz_userbiases).T) 

        Yt = np.linalg.solve(A,b)
        
        item_features[item,:] = Yt[1:num_features+1]
        item_biases[item] = Yt[0]

    return item_features, item_biases


def update_user_biased_feature(train, item_features, item_biases, lambda_user, nnz_items_per_user, nz_user_itemindices):
    
    num_users = train.shape[1]
    num_items, num_features = item_features.shape
    ones_biases = np.array([np.ones(num_items)])
    user_biases = np.zeros(num_users)
    user_features = np.zeros((num_users,num_features))
    
    for user in np.arange(num_users):        
        nnz_items = nnz_items_per_user[user]
        nz_itemindices = nz_user_itemindices[user]
        nz_itemfeatures = item_features[nz_itemindices,:]
        nz_onesbiases = ones_biases[:,nz_itemindices]
        nz_itembiases = item_biases[nz_itemindices]
        
    
        Yt = np.concatenate((nz_onesbiases, nz_itemfeatures.T), axis=0)
        A = Yt.dot(Yt.T) + lambda_user*nnz_items*np.eye(num_features+1)  
        train_user = train[nz_itemindices,user]
        b = Yt.dot(data_item_biased(train_user, nz_itembiases)) 
        Xt = np.linalg.solve(A,b)
        
        user_features[user,:] = Xt[1:num_features+1]
        user_biases[user] = Xt[0]

    return user_features, user_biases

def compute_error_prediction(data, prediction, nz):
    real_label = np.array([data[d,n] for (d,n) in nz])
    prediction_label = np.array([prediction[d,n] for (d,n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label,prediction_label))
    return rmse


def prediction_biased(item_features, item_biases, user_features, user_biases):    
    prediction_data =    user_features.dot(item_features.T).T
    prediction = ((prediction_data + user_biases).T + item_biases).T       
    return prediction


def ALS_biased(train, test, num_features = 10, lambda_user = 0.1, lambda_item = 0.1, max_it=50, seed=552):
    stop_criterion = 1e-7
    error_list = [0, 0]
    error_old = 10
    error_new = 5
    
    # set seed
    np.random.seed(seed)

    # init ALS
    user_features, item_features, user_biases, item_biases = init_MF_ALS_biased(train, num_features)
    # ***************************************************
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))    
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    _,nz_user_itemindices = map(list,zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _,nz_item_userindices = map(list,zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]

    print("learn the matrix factorization using ALS...")

    for it in np.arange(max_it):
        print("Running {} / max {} iterations".format(it+1, max_it))
        
        item_features, item_biases = update_item_biased_feature(train, user_features, user_biases, lambda_item, nnz_users_per_item, nz_item_userindices)
        user_features, user_biases = update_user_biased_feature(train, item_features, item_biases, lambda_user, nnz_items_per_user, nz_user_itemindices)
        
        prediction = prediction_biased(item_features, item_biases, user_features, user_biases)        
        train_rmse = compute_error_prediction(train, prediction, nz_train)        
        print("iter: {}, RMSE on training set: {}.".format(it+1, np.round(train_rmse,5)))
        
        error_new = compute_error_prediction(test, prediction, nz_test)
        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break
        if error_new>error_old:
            #print("Best iter: {}, with RMSE on test data: {}. ".format(it-1,error_old))
            break
        error_old = error_new

    prediction = prediction_biased(item_features, item_biases, user_features, user_biases)
    test_rmse = compute_error_prediction(test, prediction, nz_test)
    print("RMSE on test data: {}.".format(np.round(test_rmse,5)))
    return prediction, train_rmse, test_rmse, user_features, item_features
    
    # ***************************************************
    
def ridge_regression(y, tx,lambda_ = 0.1):
    """
    Least squares using normal equations (with L2 regularization)
    """
    
    reg = 2 * y.size * lambda_ * np.identity(tx.shape[1]) # L2 regularization term
    w = np.linalg.solve(tx.T.dot(tx) + reg, tx.T.dot(y))
    return w

def feature_adding(train, test, pred):
    """
    built y = real_labels  tx = (pred, #user ratings, #movie ratings, mean rate per user, mean rate per movie)
    May be also add std deviation
    
    """
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    
    nnz_u = np.copy(train)
    nnz_u[np.where(train > 0)] = 1
    
    nnz_i = np.copy(train)
    nnz_i[np.where(train > 1)] = 1
    
    num_u = nnz_u.sum(axis=0)
    num_i = nnz_i.sum(axis=1)  
    mean_u = train.sum(axis=0)/num_u
    mean_i = train.sum(axis=1)/num_i
    
    
    std_u = np.std(train, axis=0)
    std_i = np.std(train, axis=1)
    
    y = np.array([train[d,n] for (d,n) in nz_train])
    y_test = np.array([test[d,n] for (d,n) in nz_test])
    
    
    tX = np.array([[pred[d,n],num_u[n],num_i[d],mean_u[n],mean_i[d],std_u[n],std_i[d]] for (d,n) in nz_train])
    tX_test = np.array([[pred[d,n],num_u[n],num_i[d],mean_u[n],mean_i[d],std_u[n],std_i[d]] for (d,n) in nz_test])
   
    return y, y_test, tX, tX_test


def error_mse(y, tx, w):
    rmse = np.sqrt((1/len(y))*calculate_mse(y,tx.dot(w)))
    return rmse

def feature_adding_all(train, test, pred):
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    
    nnz_u = np.copy(train)
    nnz_u[np.where(train > 0)] = 1
    
    nnz_i = np.copy(train)
    nnz_i[np.where(train > 1)] = 1
    
    num_u = nnz_u.sum(axis=0)
    num_i = nnz_i.sum(axis=1)  
    mean_u = train.sum(axis=0)/num_u
    mean_i = train.sum(axis=1)/num_i
    
    
    std_u = np.std(train, axis=0)
    std_i = np.std(train, axis=1)
    
    ind =  itertools.product(np.arange(train.shape[0]), np.arange(train.shape[1]))
    
    tX = np.array([[pred[d,n],num_u[n],num_i[d],mean_u[n],mean_i[d],std_u[n],std_i[d]] for (d,n) in ind])
   
    return tX

def pred_all(tX_all, w):
    return tX_all.dot(w)

def prediction_non_biased(item_features, user_features):    
    prediction = user_features.dot(item_features.T).T    
    return prediction

def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    # ***************************************************
    num_items,num_users = train.shape
    num_features = item_features.shape[1]
    user_feature = np.zeros((num_users,num_features))
    for user in np.arange(num_users):
        nnz_items = nnz_items_per_user[user]
        nz_itemindices = nz_user_itemindices[user]
        nz_itemfeatures = item_features[nz_itemindices,:]
        A = ((nz_itemfeatures.T).dot(nz_itemfeatures)+lambda_user*nnz_items*np.eye(num_features))
        train_user = train[nz_itemindices,user]
        b = ((nz_itemfeatures.T).dot(train_user))
        user_feature[user,:] = np.linalg.solve(A,b)
    # ***************************************************
    return user_feature

def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    # ***************************************************
    num_items,num_users = train.shape
    num_features = user_features.shape[1]
    item_feature = np.zeros((num_items,num_features))
    for item in np.arange(num_items):
        nnz_users = nnz_users_per_item[item]
        nz_userindices = nz_item_userindices[item]
        nz_userfeatures = user_features[nz_userindices,:]
        A = ((nz_userfeatures.T).dot(nz_userfeatures)+lambda_item*nnz_users*np.eye(num_features))
        train_item = (train[item,nz_userindices])
        b = ((nz_userfeatures.T).dot(train_item))
        item_feature[item,:] = np.linalg.solve(A,b)
    # ***************************************************
    return item_feature


def ALS(train, test, num_features = 10, lambda_user = 0.1, lambda_item = 0.1, seed=552):
    
    stop_criterion = 1e-7

    error_list = [0, 0]
    max_it = 10 
    
    error_old = 10
    error_new = 5
    
    # set seed
    np.random.seed(seed)

    # init ALS
    user_features, item_features = init_MF_ALS(train, num_features)
    
    # ***************************************************
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))    
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    _,nz_user_itemindices = map(list,zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _,nz_item_userindices = map(list,zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]

    print("learn the matrix factorization using ALS...")

    for it in np.arange(max_it):
        print("Running {} / {} iterations".format(it+1, max_it))
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        prediction = prediction_non_biased(item_features, user_features)
        
        train_rmse = compute_error_prediction(train, prediction, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it+1, np.round(train_rmse,5)))
        
        error_new = compute_error_prediction(test, prediction, nz_test)
        error_list.append(rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break
        if error_new>error_old:
            print("Best iter: {}, with RMSE on test data: {}. ".format(it-1,error_old))
            break
        error_old = error_new
        
    prediction = prediction_non_biased(item_features, user_features)
    test_rmse = compute_error_prediction(test, prediction, nz_test)
    print("RMSE on test data: {}.".format(np.round(test_rmse,5)))

    return prediction, train_rmse, test_rmse
    # ***************************************************
