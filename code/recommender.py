import numpy as np
from helpers import *
from SGD_helpers import *
from ALS_helpers import *
from CCD_helpers import *



def matrix_factorization_SGD(train, test, num_features = 10, lambda_user = 0.1, lambda_item = 0.7, gamma = 0.01, num_epochs = 10):
    """matrix factorization by SGD."""
    
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





def matrix_factorization_SGD_b(train, test,num_features = 2,lambda_user = 0.015, lambda_item = 0.015 ,gamma = 0.05, max_it = 10, stop_criterion = 1e-7):
    """matrix factorization by SGD biased."""    
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF_b(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    b_u = mean_user(train,test)
    b_i = mean_item(train,test)
    b_g = mean_global(train,test)
      
    
    num_items, num_users = train.shape

    prediction_old = prediction_b(item_features, user_features, b_u, b_i, b_g)    
    prediction_new = np.copy(prediction_old)
    
    test_error_old = 10
    test_error_new = 5
    train_error_old = 10
    train_error_new = 5

    
        
    print("learn the matrix factorization using SGD...")
    for it in np.arange(max_it):     
      
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            
            prediction = item_features[:,d].dot(user_features[:,n])
            #gradient
            prediction_error = (train[d,n] - b_u[d,n] - b_i[d,n] - b_g[d,n] - prediction)
            
            
            #gradient entries for W
            gradient_w= -(prediction_error)*(user_features[:,n].T) + lambda_item * (item_features[:,d].T)
            #gradient entries for Z
            gradient_z = -(prediction_error)*(item_features[:,d]) + lambda_user * (user_features[:,n].T)
            
            
            b_u[d,n] += gamma*(prediction_error - lambda_user*b_u[d,n])
            b_i[d,n] += gamma*(prediction_error - lambda_item*b_i[d,n])
            
            #update
            item_features[:,d] -=  gamma*gradient_w.T
            user_features[:,n] -=  gamma*gradient_z.T
            


        prediction_new = prediction_b(item_features, user_features, b_u, b_i, b_g)          
        train_error_new = compute_error_b(train, user_features, item_features, nz_train, b_u, b_i, b_g)       
        print("iter: {}, RMSE on training set: {}.".format(it, train_error_new))
        
        test_error_new = compute_error_b(test, user_features, item_features, nz_test, b_u, b_i, b_g)
            
        if abs(train_error_old - train_error_new) < stop_criterion:
            break    
    
        if test_error_new > test_error_old:
            print("Best iter: {}, with RMSE on test data: {}. ".format(it-1,test_error_old))
            break       
        
        train_error_old = train_error_new
        test_error_old = test_error_new
        prediction_old = prediction_new   
        
        
    print("RMSE on test data: {}.".format(test_error_old))
    return prediction_old, train_error_old, test_error_old





def ALS_numpy(train, test, num_features = 5, lambda_user = 0.01, lambda_item = 0.01, max_it = 25, stop_criterion = 1e-7 ):
    """matrix factorization by ALS biased. (numpy mean that it works with numpy array for train and test"""  

    # init ALS
    user_features, item_features = init_MF_ALS_numpy(train, num_features)
    
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
    
    # initialize memory, it's to be able to return best result 
    prediction_old = prediction_non_biased(item_features, user_features)                       
    prediction_new = np.copy(prediction_old)
    
    test_error_old = 10
    test_error_new = 5
    train_error_old = 10
    train_error_new = 5
    print("learn the matrix factorization using ALS...")

    for it in np.arange(max_it):
        
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        
        prediction_new = prediction_non_biased(item_features, user_features)
        
        train_error_new = compute_error_prediction(train, prediction_new, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, train_error_new))
        
        test_error_new = compute_error_prediction(test, prediction_new, nz_test)
        
        if abs(train_error_new-train_error_old)<stop_criterion:
            break
        if test_error_new>test_error_old:
            print("Best iter: {}, with RMSE on test data: {}. ".format(it-1,test_error_old))
            break
            
        train_error_old = train_error_new
        test_error_old = test_error_new
        prediction_old = prediction_new
        
    print("RMSE on test data: {}.".format(test_error_new))
    #print("done")
    return prediction_old, train_error_old, test_error_old




def ALS_biased(train, test, num_features = 2, lambda_user = 0.01, lambda_item = 0.01, max_it=5, stop_criterion = 1e-7):
    '''matrix factorization by ALS biased.'''

    # init ALS_biased
    user_features, item_features, user_biases, item_biases = init_MF_ALS_biased(train, num_features)
    
    
    # get non-zeros indices/counts     
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))    
    
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    _,nz_user_itemindices = map(list,zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _,nz_item_userindices = map(list,zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]

    # initialize memory, it's to be able to return best result 
    prediction_old = prediction_biased(item_features, item_biases, user_features, user_biases)        
    prediction_new = np.copy(prediction_old)
    
    test_error_old = 10
    test_error_new = 5
    train_error_old = 10
    train_error_new = 5
    
    print("learn the matrix factorization using ALS biased...")

    
    for it in np.arange(max_it):
        
        
        item_features, item_biases = update_item_biased_feature(train, user_features, user_biases, lambda_item, nnz_users_per_item, nz_item_userindices)
        user_features, user_biases = update_user_biased_feature(train, item_features, item_biases, lambda_user, nnz_items_per_user, nz_user_itemindices)
        
        prediction_new = prediction_biased(item_features, item_biases, user_features, user_biases)        
        train_error_new = compute_error_prediction(train, prediction_new, nz_train)        
        print("iter: {}, RMSE on training set: {}.".format(it, train_error_new))
        
        test_error_new = compute_error_prediction(test, prediction_new, nz_test)
        
        # case improve is to small
        if abs(train_error_old - train_error_new) < stop_criterion:
            break
            
        # case overfitting
        if test_error_new > test_error_old:
            print("Best iter: {}, with RMSE on test data: {}. ".format(it-1,test_error_old))
            break
        
        train_error_old = train_error_new
        test_error_old = test_error_new
        prediction_old = prediction_new

    
    print("RMSE on test data: {}.".format(test_error_old))
    return prediction_old, train_error_old, test_error_old



 

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
    user_features, item_features = init_MF_CCD(train, num_features)

    # ***************************************************
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    _,nz_user_itemindices = map(list,zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _,nz_item_userindices = map(list,zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]
    max_it = 1e3
    
    print("learn the matrix factorization using CCD...")
    
    num_items,num_users = train.shape
    residual = train - item_features.dot(user_features)

    for it in np.arange(max_it):
        for user in np.arange(num_users):
            [residual,user_features] = update_user_CCD(residual, user_features, item_features, user, lambda_user, nnz_items_per_user, nz_user_itemindices)
            
        for item in np.arange(num_items):
            [residual,item_features] = update_item_CCD(residual, user_features, item_features, item, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        train_rmse = compute_error_residual(residual, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, train_rmse))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(test_rmse))
    
    return train_rmse, test_rmse, user_features, item_features

# Cyclic coordinate descent
def CCD_simple(train, nz_train, nz_user_itemindices, nz_item_userindices, nnz_items_per_user, nnz_users_per_item, lambda_user=0.1, lambda_item=0.7, max_it = 100):
    """Cyclic coordinate descent (CCD) algorithm."""
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init CCD
    user_features, item_features = init_MF(train, 1)
        
    # ***************************************************    
    #print("learn one feature using CCD...")
    
    num_items,num_users = train.shape
    residual = train - item_features.dot(user_features)
    
    for it in np.arange(max_it):
        for user in np.arange(num_users):
            [residual,user_features] = update_user_CCD(residual, user_features, item_features, user, lambda_user, nnz_items_per_user, nz_user_itemindices)
            
        for item in np.arange(num_items):
            [residual,item_features] = update_item_CCD(residual, user_features, item_features, item, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        train_rmse = compute_error_residual(residual, nz_train)
        #print("iter: {}, RMSE on training set: {}.".format(it, train_rmse))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break
    
    return user_features, item_features

# Feature wise update - Cyclic coordinate descent
def CCDplus(train, test, num_features=10, lambda_user=0.1, lambda_item=0.7, max_it_inter = 100):
    """Cyclic coordinate descent (CCD) algorithm."""
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init CCD++
    user_features, item_features = init_MF_CCD(train, num_features)
    
    # ***************************************************
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    nz_train, nz_row_colindices, nz_col_rowindices = build_index_groups(train)
    _,nz_user_itemindices = map(list,zip(*nz_col_rowindices))
    nnz_items_per_user = [len(i) for i in nz_user_itemindices]
    _,nz_item_userindices = map(list,zip(*nz_row_colindices))
    nnz_users_per_item = [len(i) for i in nz_item_userindices]
    max_it = 1e3
    
    print("learn the matrix factorization using CCD++...")
    
    num_items,num_users = train.shape
    residual = train - item_features.dot(user_features)

    for it in np.arange(max_it):
        for feature in np.arange(num_features):
            # one rank problem
            add = np.dot(np.c_[item_features[:,feature]],np.c_[user_features[feature,:]].T)
            feature_residual = residual + add
            
            # solving this problem using ccd
            u, v = CCD_simple(feature_residual, nz_train, nz_user_itemindices, nz_item_userindices, nnz_items_per_user, nnz_users_per_item, lambda_user, lambda_item, max_it_inter)


            # update
            user_features[feature,:] = u
            item_features[:,feature] = v.T
            
            residual = feature_residual - v.dot(u)
            
        
        train_rmse = compute_error_residual(residual, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, train_rmse))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(test_rmse))
    
    return train_rmse, test_rmse, user_features, item_features