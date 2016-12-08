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
    validation[val_ind] = Xdense[val_ind]
    train_ind = shuffled_index[np.setxor1d(range(0,data_size),range(start_val_ind,end_val_ind))]
    train[train_ind] = Xdense[train_ind]                  ## Training data indices
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


def cross_validation(ratings, K, method, num_items_per_user, num_users_per_item, min_num_ratings):
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
    for k in range(K):
        print('Running {}th fold in {} folds'.format(k+1, K))
        ### Split data in kth fold
        [training, validation] = k_fold_generator(valid_ratings, K, k, batch_size, data_size, shuffled_index)
        
        ### Matrix factorization using SGD/ALS
        if method == 0:  ## SGD
            [train_rmse, validation_rmse, user_feature, item_features] = matrix_factorization_SGD(training,
                                                 validation, num_features, lambda_user, lambda_item, gamma) 
        elif method == 1:            ## ALS
            [train_rmse, validation_rmse, user_feature, item_features] = ALS(training,
                                                    validation, num_features, lambda_user, lambda_item) 
        elif method == 2:
            [train_rmse, validation_rmse, user_feature, item_features] = CCD(training, validation, 
                                                                num_features, lambda_user, lambda_item)
        else:
            print("Incorrect method, 0-SGD, 1-ALS, 2-CCD")
            
        train_rmse_arr.append(train_rmse)
        validation_rmse_arr.append(validation_rmse)
        
    return train_rmse_arr, validation_rmse_arr