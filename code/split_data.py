import numpy as np
import scipy.sparse as sp

def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
    min_num_ratings:all users and items we keep must have at least min_num_ratings per user and per item.
    """
    # set seed
    np.random.seed(988)

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
    for i in np.arange(len(index_row)):
        # train or test
        choice = np.random.choice(a=[1, 2], p=[0.1, 0.9])
        if(choice == 1):
            test[index_row[i], index_col[i]] = valid_ratings[index_row[i], index_col[i]]
        else:
            train[index_row[i], index_col[i]] = valid_ratings[index_row[i], index_col[i]]
    # ***************************************************
    #print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    #print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    #print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test