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

def matrix_factorization_SGD_b(train, test):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 8   # K in the lecture notes =20
    lambda_user = 0.05
    lambda_item = 0.10
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
        

        #diff = train.toarray() - item_features.T.dot(user_features)

        for d, n in nz_train:
            
            prediction = item_features[:,d].dot(user_features[:,n])
            #gradient
            prediction_error = (train[d,n] - b_u[d,n] - b_i[d,n] - b_g[d,n] - prediction )
            #print(prediction_error)
            
            
            #gradient entries for W
            gradient_w= -(prediction_error)*(user_features[:,n].T) + lambda_item * (item_features[:,d].T)
            #gradient entries for Z
            gradient_z = -(prediction_error)*(item_features[:,d]) + lambda_user * (user_features[:,n].T)
            
            
            b_u[d,n] += gamma*(prediction_error - lambda_user*b_u[d,n])
            b_i[d,n] += gamma*(prediction_error - lambda_item*b_i[d,n])
            #b_g[d,n] += gamma*(prediction_error - lambda_item*lambda_user*b_g[d,n])
            
            #update
            item_features[:,d] -=  gamma*gradient_w.T
            user_features[:,n] -=  gamma*gradient_z.T

        
        rmse = compute_error_b(train, user_features, item_features, nz_train, b_u, b_i, b_g)

        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        
        error_new = compute_error_b(test, user_features, item_features, nz_test, b_u, b_i, b_g)
        print("iter: {}, RMSE on testing set: {}.".format(it, error_new))
        it += 1
        errors.append(rmse)
    rmse = compute_error_b(test, user_features, item_features, nz_test, b_u, b_i, b_g)
    print("RMSE on test data: {}.".format(rmse))
    pred = item_features.T.dot(user_features)+b_u+b_i+b_g
    return pred,nz_train,nz_test

def init_MF_ALS_biased(train, num_features, factor_features=0.1, factor_biases=1):
    ''' init all 4 factors to ones * factor '''
    num_items, num_users = train.shape
    user_features = factor_features*np.ones((num_users,num_features))
    item_features = factor_features*np.ones((num_items,num_features))
    user_biases = factor_biases*np.ones(num_users)
    item_biases = factor_biases*np.ones(num_items)
    return user_features, item_features, user_biases, item_biases


def init_MF_ALS_numpy(train, num_features, factor_features=0.1):
    ''' init all 2 factors to ones * factor '''
    num_items, num_users = train.shape
    user_features = factor_features*np.ones((num_users,num_features))
    item_features = factor_features*np.ones((num_items,num_features))
    return user_features, item_features

def data_user_biased(data, user_biases):
    ''' return data with the  user bias '''
    data_user_biased = data - user_biases
    return data_user_biased

def data_item_biased(data, item_biases):
    ''' return data with the  item bias '''
    data_item_biased = (data.T - item_biases).T
    return  data_item_biased

def compute_error_prediction(data, prediction, nz):
    ''' compute error based on the prediction and the data'''
    real_label = np.array([data[d,n] for (d,n) in nz])
    prediction_label = np.array([prediction[d,n] for (d,n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label,prediction_label))
    return rmse

def prediction_biased(item_features, item_biases, user_features, user_biases):
    ''' return prediction based on all 4 infos (normal and biases)'''
    prediction_data =    user_features.dot(item_features.T).T
    prediction = ((prediction_data + user_biases).T + item_biases).T       
    return prediction

def update_item_biased_feature(train, user_features, user_biases, lambda_item, nnz_users_per_item, nz_item_userindices):
    '''update item biased feature according to ALS biased'''
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
    '''update user biased feature according to ALS biased'''
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



def ALS_biased(train, test, num_features = 2, lambda_user = 0.01, lambda_item = 0.01, max_it=25, stop_criterion = 1e-7):
    '''als biased version'''

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


def ALS_numpy(train, test, num_features = 5, lambda_user = 0.01, lambda_item = 0.01, max_it = 25, stop_criterion = 1e-7 ):
    

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
    item_features = np.ones((num_items,num_features))
    user_features = np.ones((num_features,num_users))
    # ***************************************************
    return 1.0*user_features,1.0*item_features

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

def init_MF_CCD_biased(train, num_features, factor_features=0.1, factor_biases=1):
    num_items, num_users = train.shape
    user_features = factor_features*np.ones((num_users,num_features))
    item_features = factor_features*np.ones((num_items,num_features))
    user_biases = factor_biases*np.array(train.sum(axis=0)/train.getnnz(axis=0)) #np.ones(num_users)
    item_biases = factor_biases*np.array(train.sum(axis=1)/train.getnnz(axis=1)) #np.ones(num_items)
    return user_features, item_features, user_biases, item_biases

def data_user_biased(data, user_biases):
    data_user_biased = data - user_biases
    return data_user_biased
def data_item_biased(data, item_biases):
    data_item_biased = (data.T - item_biases).T
    return  data_item_biased

def prediction_biased(item_features, item_biases, user_features, user_biases):    
    prediction_data =    user_features.dot(item_features.T).T
    prediction = ((prediction_data + user_biases).T + item_biases).T       
    return prediction

# Cyclic coordinate descent
def CCD_simple_biased(train, nz_train, user_biases, item_biases, nz_user_itemindices, nz_item_userindices, nnz_items_per_user, nnz_users_per_item, lambda_user=0.1, lambda_item=0.7, max_it = 100):
    """Cyclic coordinate descent (CCD) algorithm."""
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init CCD
    user_features, item_features = init_MF_CCD_biased(train, 1)
        
    # ***************************************************    
    #print("learn one feature using CCD...")
    
    num_items,num_users = train.shape
    residual = train - item_features.dot(user_features)
    
    for it in np.arange(max_it):
        for user in np.arange(num_users):
            [residual,user_features] = update_user_CCD_biased(residual, user_features, item_features, user, lambda_user, nnz_items_per_user, nz_user_itemindices)
            
        for item in np.arange(num_items):
            [residual,item_features] = update_item_CCD_biased(residual, user_features, item_features, item, lambda_item, nnz_users_per_item, nz_item_userindices)
        
        train_rmse = compute_error_residual(residual, nz_train)
        #print("iter: {}, RMSE on training set: {}.".format(it, train_rmse))        
        error_list.append(train_rmse)
        if abs(error_list[-1]-error_list[-2])<stop_criterion:
            break
    
    return user_features, item_features

# Feature wise update - Cyclic coordinate descent biased
def CCDplus_biased(train, test, num_features=10, lambda_user=0.1, lambda_item=0.7, max_it_inter = 100):
    """Cyclic coordinate descent (CCD) algorithm."""
    # define parameters
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]
    
    # set seed
    np.random.seed(988)

    # init CCD++
    user_features, item_features, user_biases, item_biases = init_MF_CCD_biased(train, num_features)
    
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
    residual = train - prediction_biased(item_features, item_biases, user_features, user_biases)

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