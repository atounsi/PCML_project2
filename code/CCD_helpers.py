import numpy as np
from helpers import *

def update_user_CCD(
        residual, user_features, item_features, user, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature row."""
    # ***************************************************
    new_residual = residual
    new_user_f = user_features
    num_items,num_users = residual.shape
    num_features = item_features.shape[1]
    nnz_items = nnz_items_per_user[user]
    nz_itemindices = nz_user_itemindices[user]
    for t in np.arange(num_features):
        nom = new_residual[nz_itemindices,user]+ new_user_f[t,user]*np.c_[item_features[nz_itemindices,t]]
        nom = (item_features[nz_itemindices,t].T).dot(nom)[0,0]
        denom = lambda_user + (item_features[nz_itemindices,t].T).dot(item_features[nz_itemindices,t])
        new = nom/denom
        new_res = new_residual[nz_itemindices,user] - (new-new_user_f[t,user])*np.c_[item_features[nz_itemindices,t]]
        new_residual[nz_itemindices,user] = np.squeeze(np.asarray(new_res))
        new_user_f[t,user] = new
    # ***************************************************
    return new_residual , new_user_f

def update_item_CCD(
        residual, user_features, item_features, item, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature column."""
    # ***************************************************
    new_residual = residual
    new_item_f = item_features
    num_items,num_users = residual.shape
    num_features = user_features.shape[0]
    nnz_users = nnz_users_per_item[item]
    nz_userindices = nz_item_userindices[item]
    for t in np.arange(num_features):
        nom = new_residual[item,nz_userindices] + new_item_f[item,t]*np.r_[user_features[t,nz_userindices]]
        nom = nom.dot((user_features[t,nz_userindices]).T)[0,0]
        denom = lambda_item + (user_features[t,nz_userindices]).dot(user_features[t,nz_userindices].T)
        new = nom/denom
        new_res = new_residual[item,nz_userindices] - (new-new_item_f[item,t])*user_features[t,nz_userindices]
        new_residual[item,nz_userindices] = np.squeeze(np.asarray(new_res))
        new_item_f[item,t] = new
    # ***************************************************
    return new_residual , new_item_f

def compute_error_residual(residual, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    t = np.array([residual[d,n] for (d,n) in nz])
    rmse = np.sqrt(t.dot(t.T)/len(nz))
    # ***************************************************
    return rmse

def init_MF_CCD(train, num_features):
    """init the parameter for matrix factorization."""
    
    # return init_MF(train, num_features)
    # return init_MF_ALS(train, num_features)
    # ***************************************************
    num_items, num_users = train.shape
    user_mean = np.array(train.sum(axis=0)/train.getnnz(axis=0))
    user_features = np.r_[user_mean,np.random.randn(num_features-1,num_users)]
    item_mean = np.array(train.sum(axis=1).T/train.getnnz(axis=1)).T
    item_features = np.c_[item_mean,item_mean+np.random.randn(num_items,num_features-1)]
    #item_features = np.zeros((num_items,num_features))
    # ***************************************************
    return 1.0*user_features,1.0*item_features

def init_MF_CCD_simple(train, num_features):
    """init the parameter for matrix factorization."""
    
    # return init_MF(train, num_features)
    # return init_MF_ALS(train, num_features)
    # ***************************************************
    num_items, num_users = train.shape
    item_features = np.zeros((num_items,num_features))
    user_features = np.zeros((num_features,num_users))
    # ***************************************************
    return 1.0*user_features,1.0*item_features