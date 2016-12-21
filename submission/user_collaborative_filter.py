import numpy as np
import scipy.sparse as sp
from helpers import *
from split_data import *

def compute_error_prediction(data, prediction, nz):
    real_label = np.array([data[d,n] for (d,n) in nz])
    prediction_label = np.array([prediction[d,n] for (d,n) in nz])
    rmse = np.sqrt((1/len(nz))*calculate_mse(real_label,prediction_label))
    return rmse

def user_collaborative_filter(train, test, num_user_neighbours=50, nosubmit=0):
    user_mean = train.sum(axis=0) / sp.lil_matrix(train).getnnz(axis=0)
    sim_user = np.loadtxt(open("sim_user.csv", "rb"), delimiter=",", skiprows=0)

    if not nosubmit:
        path_dataset = "data/sampleSubmission.csv"
        submission_ratings = load_data(path_dataset)
        nz_row, nz_col = submission_ratings.nonzero()
    else:    
        nz_row, nz_col = test.nonzero()

    missing_entries = list(zip(nz_row, nz_col))
    prediction_usercollab = np.zeros(test.shape)

    print("Running user collaborative filtering...")
    for i in range(len(missing_entries)):
        item_to_predict = missing_entries[i][0]
        user_to_predict = missing_entries[i][1]
    
        users_seen_item = train[item_to_predict,:].nonzero()
        s = abs(sim_user[users_seen_item, user_to_predict][0])
        ind = sorted(range(len(s)), reverse=True, key=lambda k: s[k])
    
        user_neighbours = users_seen_item[0][ind][:num_user_neighbours]
    
        if len(user_neighbours)<num_user_neighbours:
            prediction_usercollab[item_to_predict, user_to_predict] = user_mean[user_to_predict]
        else:
            agg = 0
            sim_user_agg = 0
            for j in range(num_user_neighbours):
                neigbhour = user_neighbours[j] 
                agg += sim_user[neigbhour, user_to_predict] * (train[item_to_predict, neigbhour] - user_mean[neigbhour]) 
                sim_user_agg += abs(sim_user[neigbhour, user_to_predict])

            user_collab_filt_pred = user_mean[user_to_predict] + agg/sim_user_agg
            prediction_usercollab[item_to_predict, user_to_predict] = user_collab_filt_pred

    if nosubmit:
        nz_row, nz_col = test.nonzero()
        nz_test = list(zip(nz_row, nz_col))
        test_rmse = compute_error_prediction(test, prediction_usercollab, nz_test)
        print("RMSE on test data: {}.".format(np.round(test_rmse,5)))

    return prediction_usercollab
