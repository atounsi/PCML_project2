def preprocess(X):
    """
    Preprocess the ratings
    """
    return X

def remove_user_mean(train):
    """
    need nnz train
    """
    
    num_u = train.getnnz(axis=0)
    mean_u = np.array(train.sum(axis=0)/num_u)[0]
    
    return train - mean
    


# linear classifier on top of the prediction
def feature_adding(train, test, item_features, user_features):
    """
    built y = real_labels  tx = (pred, #user ratings, #movie ratings, mean rate per user, mean rate per movie)
    
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