import numpy as np
import itertools
from helpers import *

def feature_adding(train, test, pred):
    """
    built y = real_labels  tx = (pred, #user ratings, #movie ratings, mean rate per user, mean rate per movie)
    May be also add std deviation
    
    """
    
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    
    nnz_u = np.copy(train)
    nnz_u[np.where(train > 0)] = 1
    
    nnz_i = np.copy(train)
    nnz_i[np.where(train > 1)] = 1
    
    num_u = nnz_u.sum(axis=0)
    num_i = nnz_i.sum(axis=1)  
    mean_u = train.sum(axis=0)/num_u
    mean_i = train.sum(axis=1)/num_i
    
    
    std_u = np.std(train, axis=0)
    std_i = np.std(train, axis=1)
    
    y = np.array([train[d,n] for (d,n) in nz_train])
    y_test = np.array([test[d,n] for (d,n) in nz_test])
    
    
    tX = np.array([[pred[d,n],num_u[n],num_i[d],mean_u[n],mean_i[d],std_u[n],std_i[d]] for (d,n) in nz_train])
    tX_test = np.array([[pred[d,n],num_u[n],num_i[d],mean_u[n],mean_i[d],std_u[n],std_i[d]] for (d,n) in nz_test])
   
    return y, y_test, tX, tX_test


#least square
def error_mse(y, tx, w):
    rmse = np.sqrt((1/len(y))*calculate_mse(y,tx.dot(w)))
    return rmse

def ridge_regression(y, tx,lambda_ = 0.1):
    """
    Least squares using normal equations (with L2 regularization)
    """
    
    reg = 2 * y.size * lambda_ * np.identity(tx.shape[1]) # L2 regularization term
    w = np.linalg.solve(tx.T.dot(tx) + reg, tx.T.dot(y))
    return w

def build_poly(x, degree):
    """
    Build a polynomial basis function for input data x.
    The polynomial is built independently for each feature, e.g.
    for degree = 3 and D = 3, the basis is:
    x x^2 x^3   y y^2 y^3   z z^2 z^3
    Therefore, features are not combined together.
    """
    # The first feature is not transformed (bias added by "standardize")
    first = x[:,0].reshape(x.shape[0], 1) # Constant bias
    x_ = np.delete(x, 0, axis=1)
    
    A = np.repeat(x_, degree, axis=1)
    B = np.tile(np.arange(1, degree + 1), [x_.shape[0], x_.shape[1]])
    
    return np.concatenate((first, np.power(A, B)), axis=1)



def feature_adding_all(train, test, pred):
    
    data = train + test
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    
    
    nnz_u = np.copy(data)
    nnz_u[np.where(data > 0)] = 1
    
    nnz_i = np.copy(data)
    nnz_i[np.where(data > 1)] = 1
    
    num_u = nnz_u.sum(axis=0)
    num_i = nnz_i.sum(axis=1)  
    mean_u = data.sum(axis=0)/num_u
    mean_i = data.sum(axis=1)/num_i
    
    
    std_u = np.std(data, axis=0)
    std_i = np.std(data, axis=1)
    
    ind =  itertools.product(np.arange(data.shape[0]), np.arange(data.shape[1]))
    
    tX = np.array([[pred[d,n],num_u[n],num_i[d],mean_u[n],mean_i[d],std_u[n],std_i[d]] for (d,n) in ind])
   
    return tX

def pred_all(tX_all, w):
    return tX_all.dot(w)


def linear_corrector(pred, train, test, lambda_ = 0.015, degree = 5 ):
    
    y, y_test, tX, tX_test = feature_adding(train, test, pred)
    print("features added to train and test")
    
    #colud be looped over degree
    tX_poly = build_poly(tX, degree)
    tX_test_poly = build_poly(tX_test, degree)
    print("polynomial version built")
    
    #colud be looped over lamdas
    w = ridge_regression(y, tX_poly, lambda_)
    print("train error after ridge regression:")
    print(error_mse(y, tX_poly, w))
    print("test error after ridge regression:")
    print(error_mse(y_test, tX_test_poly, w))
    
    
    #take lot of memory and time (2min)
    tX_all = feature_adding_all(train, test, pred)
    print("features added to to whole data")
    tX_all_poly = build_poly(tX_all, degree)
    print("polynomial version built")
    
    pred_final = pred_all(tX_all_poly, w)
    pred_ready = pred_final.reshape((pred.shape[0], pred.shape[1]))

    return pred_ready

def bound_corrector(pred):
    
    prediction_label = pred.reshape((pred.shape[0]* pred.shape[1],1))
    
    too_much = np.where(prediction_label > 5)
    #print("num value > 5:")
    #print(len(too_much[0]))
    not_enough = np.where(prediction_label < 1)
    #print("num value < 1:")
    #print(len(not_enough[0]))
    prediction_label[too_much]=5
    prediction_label[not_enough]=1
    corrected_pred = prediction_label.reshape((pred.shape[0], pred.shape[1]))
    
    return corrected_pred
    
    
