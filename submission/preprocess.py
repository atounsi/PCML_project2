def preprocess(X):
    """
    Preprocess the ratings
    """
    return X

    


# linear classifier on top of the prediction
def feature_adding(train, test, item_features, user_features):
    """
    built y = real_labels  tx = (pred, #user ratings, #movie ratings, mean rate per user, mean rate per movie)
    May be also add std deviation
    
    """
    
    num_u = train.getnnz(axis=0)
    num_i = train.getnnz(axis=1)   
    mean_u = np.array(train.sum(axis=0)/num_u)[0]
    mean_i = np.array(train.sum(axis=1).T/num_i)[0]

    y = np.array([train[d,n] for (d,n) in nz_train])
    y_test = np.array([test[d,n] for (d,n) in nz_test])
    

    
    tX = np.array([[(np.dot(item_features[:,d],(user_features[:,n]))),num_u[n],num_i[d],mean_u[n],mean_i[d]] for (d,n) in nz_train])
    tX_test = np.array([[(np.dot(item_features[:,d],(user_features[:,n]))),num_u[n],num_i[d],mean_u[n],mean_i[d]] for (d,n) in nz_test])
   
    return y, y_test, tX, tX_test

#least square
def least_squares(y, tx):
    """
    Least squares using normal equations.
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    return w

def error_mse(y, tx, w):
    rmse = np.sqrt((1/len(y))*calculate_mse(y,tx.dot(w)))
    return rmse


## sgd with bias work quite good on ex10 data

def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    num_items, num_users = train.shape
    
    user_features = np.ones ((num_features, num_users)) 
    item_features = np.ones ((num_features, num_items)) 

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
    num_features = 4   # K in the lecture notes =20
    lambda_user = 0.01
    lambda_item = 0.01
    num_epochs = 10     # number of full passes through the train set
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
    

    
        
    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
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
            
            #update
            item_features[:,d] -=  gamma*gradient_w.T
            user_features[:,n] -=  gamma*gradient_z.T

        
        rmse = compute_error_b(train, user_features, item_features, nz_train, b_u, b_i, b_g)

        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)
    rmse = compute_error_b(test, user_features, item_features, nz_test, b_u, b_i, b_g)
    print("RMSE on test data: {}.".format(rmse))
    return user_features, item_features,nz_train,nz_test