import numpy as np
import scipy.sparse as sp

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1,seed = 998):
    """split the ratings to training data and test data.
    Args:
    min_num_ratings:all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(seed)

    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][:, valid_users]
    
    # ***************************************************
    # NOTE: we only consider users and movies that have more than min_num_ratings
    # generate random indices
    train = sp.lil_matrix((len(valid_items), len(valid_users)))
    test = sp.lil_matrix((len(valid_items), len(valid_users)))

    index_row, index_col = valid_ratings.nonzero()
    
    choices = np.random.choice(a=[1, 2], size = len(index_row), p=[p_test, 1-p_test])
    
    for i in np.arange(len(index_row)):
        # train or test
        if(choices[i] == 1):
            test[index_row[i], index_col[i]] = valid_ratings[index_row[i], index_col[i]]
        else:
            train[index_row[i], index_col[i]] = valid_ratings[index_row[i], index_col[i]]
    # ***************************************************
    #print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    #print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    #print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test

def split_data_numpy(ratings, p_test=0.1, seed=45):
    '''Same as split_data but numpy style
    
    Returns train and test data.    
    '''
    
    # set seed
    np.random.seed(seed)
    
    # generate random indices
    row, col = ratings.shape
    num = row*col
    
    indices = np.random.permutation(num)
    
    #split and share indices between train and test
    split_index = int(np.floor(p_test * num))    
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]  
    
    
    rat_arr = ratings.toarray()
    
    # reshaping put all in 1-D
    test = np.copy(rat_arr).reshape((num,1))
    train = np.copy(rat_arr).reshape((num,1))  
    
    # create split
    train[train_indices] = 0
    test[test_indices] = 0
    
    # going back to 2-D
    train = train.reshape((row,col))    
    test = test.reshape((row,col))
    
    return train, test