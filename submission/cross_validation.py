import numpy as np
import scipy.sparse as sp
from mf_als import *
from mf_ccd import *
from mf_sgd import *
from ensemble_als_userfilter import *



def split_data_k_numpy(ratings, k_indices_set, k):
    """split the data and return numpy array of train and test"""
    
    K, index_split = k_indices_set.shape
    index_tr = k_indices_set[np.where(np.arange(K) != (k-1))].reshape(((K-1)*index_split,1))[:,0]
    index_te = k_indices_set[k-1]
    
    row = ratings.shape[0]
    col = ratings.shape[1]
    num = row*col
    
    rat_arr = ratings.toarray()
    
    # reshaping
    valid_array_te = np.copy(rat_arr).reshape((num,1))
    valid_array_tr = np.copy(rat_arr).reshape((num,1))  
    
    # create split
    train = valid_array_tr
    train[index_te] = 0
    train = train.reshape((row,col))
    
    test = valid_array_te
    test[index_tr] = 0
    test = test.reshape((row,col))
    
    # ***************************************************
    
    return train, test



def k_indices_set_generator(ratings, k=5, seed=48):
    """return K different set of indices from that whole dataset, used for cross_validation""" 
    # ***************************************************
    # set seed
    np.random.seed(seed)
    
    # generate random indices
    row = ratings.shape[0]
    col = ratings.shape[1]
    num = row*col
    
    indices = np.random.permutation(num)
    
    index_split = int(num/k)
    k_indices_set = np.zeros((k,index_split),dtype=np.int)   
    
    for i in range(k):    
        id_start = index_split*i
        id_end = index_split*(i+1)
        k_indices_set[i] = indices[id_start:id_end]
    # ***************************************************
    return k_indices_set

def split_data_k(ratings, k_indices_set, k):
    """split the data according to k_indices_set ( containing K different set of indices)"""
    # ***************************************************
    K, index_split = k_indices_set.shape
    index_tr = k_indices_set[np.where(np.arange(K) != (k-1))].reshape(((K-1)*index_split,1))[:,0]
    index_te = k_indices_set[k-1]
    
    row = ratings.shape[0]
    col = ratings.shape[1]
    num = row*col
    
    rat_arr = ratings.toarray()
    
    # reshaping
    valid_array_te = np.copy(rat_arr).reshape((num,1))
    valid_array_tr = np.copy(rat_arr).reshape((num,1))  
    
    # create split
    train = valid_array_tr
    train[index_te] = 0
    train = train.reshape((row,col))
    
    test = valid_array_te
    test[index_tr] = 0
    test = test.reshape((row,col))
    
    # ***************************************************
    return sp.lil_matrix(train), sp.lil_matrix(test)

def cross_validation_run(ratings, method, K, num_features=5, lambda_user=0.01, lambda_item=0.01, gamma = 0.01, max_iter=10,
                         lambda_ridge=0.01, weight_als=0.75, num_user_neighbours=50, nosubmit=1):
    """run a cross validation algorithm, return errors, but also best train/test for the function requested"""
    
    k_indices_set = k_indices_set_generator(ratings,K)    

    train_rmse_arr=[]
    validation_rmse_arr=[]
        
    for k in range(K):
        print('Running {}th fold in {} folds'.format(k+1, K))
        train_cross,test_cross = split_data_k(ratings, k_indices_set, k+1)

        if method == 0:     ## Ensemble (ALS + User collaborative filtering)
            [pred, validation_rmse, user_feature, item_features] = ensemble_als_userfilter(ratings, train_cross, test_cross,
                                                                num_features, lambda_user, lambda_item, max_iter, 
                                                                lambda_ridge, weight_als, num_user_neighbours, nosubmit)    
            train_rmse = [] 
        elif method == 1:   ## ALS
            [prediction, train_rmse, validation_rmse, user_feature, item_features] = ALS_biased(train_cross.toarray(),
                                                    test_cross.toarray(), num_features, lambda_user, lambda_item, max_iter) 
        elif method == 2:   ## CCD
            [train_rmse, validation_rmse, user_feature, item_features] = CCD(train_cross, test_cross, 
                                                                num_features, lambda_user, lambda_item)
        elif method == 3:   ## CCD++
            [train_rmse, validation_rmse, user_feature, item_features] = CCDplus(train_cross, test_cross, 
                                                                num_features, lambda_user, lambda_item)
        elif method == 5:   ## SGD
            [pred, train_rmse, validation_rmse, user_feature, item_features] = matrix_factorization_SGD(train_cross,
                                                 test_cross, num_features, lambda_user, lambda_item, gamma, max_iter)
        else:
            print("Incorrect method, 0-Ensemble, 1-ALS, 2-CCD, 3-CCD++, 5-SGD")

        train_rmse_arr.append(train_rmse)
        validation_rmse_arr.append(validation_rmse)

        if ((k == 1) or (k>1 and validation_rmse_arr[-1]<vr_min)) :
            us_ft_min = user_feature
            itm_ft_min = item_features
            vr_min = validation_rmse
            tr_min = train_rmse
            train = train_cross
            test = test_cross
        
    return train_rmse_arr, validation_rmse_arr, train, test, tr_min, vr_min, us_ft_min, itm_ft_min


