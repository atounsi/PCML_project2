import numpy as np
import scipy
import scipy.sparse as sp
from recommender import *

def k_fold_generator(X, K, kth_fold, batch_size, data_size, shuffled_index):   
    # Select validation and training data in kth fold
    start_val_ind = kth_fold*batch_size
    end_val_ind   = (kth_fold+1)*batch_size
    
    ##[TODO: Try to improve runtime of this method]
        
    ## ==== Dense matrix approach: memory intensive === ##
    validation = np.zeros(X.get_shape())   ## Initialise dense matrix
    train = np.zeros(X.get_shape())        ## Initialise dense matrix
    Xdense = X.todense()                   ## Change ratings to dense matrix
    
    val_ind = shuffled_index[start_val_ind:end_val_ind]   ## Validation data indices
    #validation[val_ind] = Xdense[val_ind]  ## Incorrect
    for i in range(len(val_ind)):
        validation[val_ind[i][0], val_ind[i][1]] = Xdense[val_ind[i][0], val_ind[i][1]]
        
    train_ind = shuffled_index[np.setxor1d(range(0,data_size),range(start_val_ind,end_val_ind))]
    #train[train_ind] = Xdense[train_ind]   ## Incorrect
    for i in range(len(train_ind)):         ## Training data indices
        train[train_ind[i][0], train_ind[i][1]] = Xdense[train_ind[i][0], train_ind[i][1]]
        
    return sp.lil_matrix(train), sp.lil_matrix(validation)## Return sparse matrices
    
    """
    ## ==== Sparse matrix approach: time consuming === ##
    validation = sp.lil_matrix(X.get_shape())  ## Initialise sparse matrix
    train = sp.lil_matrix(X.get_shape())       ## ## Initialise sparse matrix
    val_ind = shuffled_index[start_val_ind:end_val_ind]   ## Validation data indices
    #validation[val_ind] = X[val_ind] #[This does not work, will help improve speed]
    for ind in val_ind:
        validation[ind] = X[ind]       
    train_ind = shuffled_index[np.setxor1d(range(0,data_size),range(start_val_ind,end_val_ind))]
    #train[train_ind] = X[train_ind] #[This does not work, will help improve speed]
    for ind in train_ind:                                 ## Training data indices
        train[ind] = X[ind]           
    return train, validation
    """


def cross_validation(ratings, K, method, num_items_per_user, num_users_per_item, min_num_ratings, num_features=1, lambda_user=0.1, lambda_item=0.7, gamma=0.01):
    '''
    method: 0-SGD, 1-ALS, 2-CCD
    '''
    # set seed
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]
    
    row, col = valid_ratings.nonzero()
    ind = list(zip(row, col))
    shuffled_index = np.random.permutation(ind)
    
    batch_size = int(np.floor(len(ind)/K))
    data_size = batch_size*K
    
    train_rmse_arr=[]
    validation_rmse_arr=[]
        
    for k in range(K):
        print('Running {}th fold in {} folds'.format(k+1, K))
        ### Split data in kth fold
        [training, validation] = k_fold_generator(valid_ratings, K, k, batch_size, data_size, shuffled_index)
        
        ### Matrix factorization using SGD/ALS/CCD
        if method == 0:  ## SGD
            [train_rmse, validation_rmse, user_feature, item_features] = matrix_factorization_SGD(training,
                                                 validation, num_features, lambda_user, lambda_item, gamma) 
        elif method == 1:  ## ALS
            [train_rmse, validation_rmse, user_feature, item_features] = ALS(training,
                                                    validation, num_features, lambda_user, lambda_item) 
        elif method == 2:
            [train_rmse, validation_rmse, user_feature, item_features] = CCD(training, validation, 
                                                                num_features, lambda_user, lambda_item)
        elif method == 3:
            [train_rmse, validation_rmse, user_feature, item_features] = CCDplus(training, validation, 
                                                                num_features, lambda_user, lambda_item)
        else:
            print("Incorrect method, 0-SGD, 1-ALS, 2-CCD")
        
        train_rmse_arr.append(train_rmse)
        validation_rmse_arr.append(validation_rmse)
        
    return train_rmse_arr, validation_rmse_arr

######### lucaz version

#need k be multiple of row*col
def k_indices_set_generator(ratings, k=5, seed=48):
    
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
    
    return k_indices_set

def split_data_k(ratings, k_indices_set, k):
    
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
    
    #print (valid_ratings, train, test)
    return train, test

def split_data_k(ratings, k_indices_set, k):
    
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
    
    #print (valid_ratings, train, test)
    return sp.lil_matrix(train), sp.lil_matrix(test)

def cross_validation_minimalist(ratings, method, K, num_features=5, lambda_user=0.01, lambda_item=0.01, gamma = 0.01):
    
    k_indices_set = k_indices_set_generator(ratings,K)    

    train_rmse_arr=[]
    validation_rmse_arr=[]
        
    for k in range(K):
        print('Running {}th fold in {} folds'.format(k+1, K))
        train_cross,test_cross = split_data_k(ratings, k_indices_set, k+1)

        ### Matrix factorization using SGD/ALS/CCD
        if method == 0:  ## SGD
            [train_rmse, validation_rmse, user_feature, item_features] = matrix_factorization_SGD(train_cross,
                                                 test_cross, num_features, lambda_user, lambda_item, gamma) 
        elif method == 1:  ## ALS
            [train_rmse, validation_rmse, user_feature, item_features] = ALS(train_cross,
                                                    test_cross, num_features, lambda_user, lambda_item) 
        elif method == 2:
            [train_rmse, validation_rmse, user_feature, item_features] = CCD(train_cross, test_cross, 
                                                                num_features, lambda_user, lambda_item)
        elif method == 3:
            [train_rmse, validation_rmse, user_feature, item_features] = CCDplus(train_cross, test_cross, 
                                                                num_features, lambda_user, lambda_item)
        else:
            print("Incorrect method, 0-SGD, 1-ALS, 2-CCD")
        train_rmse_arr.append(train_rmse)
        validation_rmse_arr.append(validation_rmse)
        
    return train_rmse,validation_rmse_arr