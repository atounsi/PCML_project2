import numpy as np
from helpers import *

def init_MF_CCD(train, num_features):
    """init the parameter for matrix factorization."""
    # ***************************************************
    num_items, num_users = train.shape
    user_mean = np.array(train.sum(axis=0)/train.getnnz(axis=0))
    user_features = np.r_[user_mean,np.random.randn(num_features-1,num_users)]
    item_mean = np.array(train.sum(axis=1).T/train.getnnz(axis=1)).T
    item_features = np.c_[item_mean,item_mean+np.random.randn(num_items,num_features-1)]
    #item_features = np.zeros((num_items,num_features))
    # ***************************************************
    return 1.0*user_features,1.0*item_features

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

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    # ***************************************************
    real_label = np.array([data[d, n] for (d, n) in nz])
    prediction = np.array([(np.dot(item_features[d, :], (user_features[:, n]))) for (d, n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label, prediction))
    # ***************************************************
    return rmse


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
        print("Running {} / max {} iterations".format(int(it+1), int(max_it)))

        for user in np.arange(num_users):
            [residual,user_features] = update_user_CCD(residual, user_features, item_features, user, lambda_user, nnz_items_per_user, nz_user_itemindices)
            
        for item in np.arange(num_items):
            [residual,item_features] = update_item_CCD(residual, user_features, item_features, item, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        train_rmse = compute_error_residual(residual, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(int(it), np.round(train_rmse,5)))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(np.round(test_rmse,5)))
    
    return train_rmse, test_rmse, user_features, item_features


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
        print("Running {} / max {} iterations".format(int(it+1), int(max_it)))

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
        print("iter: {}, RMSE on training set: {}.".format(int(it), np.round(train_rmse,5)))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break

    test_rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(np.round(test_rmse,5)))
    
    return train_rmse, test_rmse, user_features, item_features

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

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""  
    # ***************************************************
    num_items, num_users = train.shape
    user_features = np.random.randint(low=0, high=5, size=(num_features, num_users))
    item_features = np.random.randint(low=0, high=5, size=(num_items, num_features))
    # ***************************************************
    return 1.0*user_features, 1.0*item_features
