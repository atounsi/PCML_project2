import numpy as np
from helpers import *

def init_MF(train, num_features):
    """init the parameter for matrix factorization, at random between 1 and 5"""  
    # ***************************************************
    num_items, num_users = train.shape
    user_features = np.random.randint(low=1, high=5, size=(num_features, num_users))
    item_features = np.random.randint(low=1, high=5, size=(num_items, num_features))
    # ***************************************************
    return 1.0*user_features, 1.0*item_features


def compute_error(data, user_features, item_features, nz):
    """compute the loss (RMSE) of the prediction (non-biased version) of nonzero elements."""
    # ***************************************************
    real_label = np.array([data[d, n] for (d, n) in nz])
    prediction = np.array([(np.dot(item_features[d, :], (user_features[:, n]))) for (d, n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label, prediction))
    # ***************************************************
    return rmse


def init_MF_b(train, num_features):
    """init the parameter for matrix factorization."""
    # ***************************************************
    num_items, num_users = train.shape    
    user_features = 0.01*np.ones((num_features, num_users)) 
    item_features = 0.01*np.ones((num_features, num_items)) 
    # ***************************************************
    return user_features, item_features


def compute_error_b(data, user_features, item_features, nz, b_u, b_i, b_g):
    """compute the loss (MSE) of the prediction (biased version) of nonzero elements."""
    # ***************************************************
    real_label = np.array([data[d,n] for (d,n) in nz])
    pred_array = np.dot(item_features.T,user_features) + b_u + b_i + b_g
    prediction = np.array([pred_array[d,n] for (d,n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label,prediction))
    # ***************************************************
    return rmse

def user_mean(data):
    """compute the ratings mean for each user"""
    # ***************************************************
    nnz_u = np.copy(data)
    nnz_u[np.where( data > 0 )] = 1
    # ***************************************************
    return data.sum(axis=0) / nnz_u.sum(axis=0)


def item_mean(data):    
    """compute the ratings mean for each item"""
    # ***************************************************
    nnz_i = np.copy(data)
    nnz_i[np.where( data > 0 )] = 1
    # ***************************************************
    return data.sum(axis=1) / nnz_i.sum(axis=1)

def global_mean(data):
    """compute the global ratings mean"""
    # ***************************************************
    nnz_g = np.copy(data)
    nnz_g[np.where( data > 0 )] = 1
    # ***************************************************
    return data.sum()/nnz_g.sum()
    

def mean_user(train,test):  
    """return a matrix with the same shape as the data, based on user mean """
    # ***************************************************
    num_items, num_users = train.shape
    mean =  user_mean(train)  
    mean_u = np.ones((num_items,num_users))*mean.T
    # ***************************************************
    return mean_u

def mean_item(train,test):  
    """return a matrix with the same shape as the data, based on item mean """
    # ***************************************************
    num_items, num_users = train.shape
    mean = item_mean(train)  
    mean_i = (np.ones((num_items,num_users)).T*mean).T
    # ***************************************************
    return mean_i

def mean_global(train,test):
    """return a matrix with the same shape as the data, based on global mean """
    # ***************************************************
    num_items, num_users = train.shape
    mean = global_mean(train)
    mean_g = np.ones((num_items,num_users))*mean
    # ***************************************************
    return mean_g

def prediction_b(item_features, user_features, b_u, b_i, b_g):
    """compute the prediction matrix based on the biases"""
    return item_features.T.dot(user_features)+b_u+b_i+b_g
