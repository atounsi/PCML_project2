###################################################
This README file describes the following:
==== Python module dependencies
==== Running run.py
==== Brief description of our Recommender system
==== Contents of zip file and File organisation
####################################################


####################################
#### Python modules requirements ###
####################################
==== Python modules: 
======= 1. numpy
======= 2. scipy
======= 3. argparse
======= 4. itertools
==== External libraries: None
####################################



###################################################################################################
########################################### Running run.py ########################################
###################################################################################################

run.py is the script to generate .csv predictions submitted to Kaggle and also to run cross validation.

================== To generate predictions ===============================
=== Command to generate .csv predictions is python run.py method_id
=== where method_id can take one of the following values 
====== 0-Ensemble (best prediction submitted to Kaggle) (ALS + User collaborative filtering)
====== 1-ALS
====== 2-CCD
====== 3-CCD++ 
====== 4-UserCollaborativeFiltering 
====== 5-SGD
=== As shown above, run.py can be used to run different methods.
=== The best results on Kaggle can be obtained by running method 0, 
    using the following command: python run.py 0
==========================================================================

================== Running cross validation ================================
=== run.py can also be used to run cross validation for all the methods
=== Command to run cross validation: python run.py {method_id} --nosubmit
=== Cross validation can be enabled by setting cross_validation_enabled=1
============================================================================

================== Varying parameters ====================================================
The parameters are by default set to the optimal values as determined by cross validation.
Following is a list of variables which can be set in run.py
=== num_features        = 2  		# Number of latent features in matrix factorization model
=== lambda_user         = 0.01 		# Regularisation parameter for user feature
=== lambda_item         = 0.01      # Regularisation parameter for item feature
=== gamma               = 0.01      # Learning rate (only for SGD)
=== num_user_neighbours = 50        # Number of users in neighbourhood (for user collaborative filtering)
=== lambda_ridge        = 0.01		# Regularisation parameter for ridge regression
=== weight_als          = 0.75      # Weight for producing ensemble prediction
=== max_iter            = 10        # Maximum number of iterations of training 
=== K                   = 10        # K-fold cross validation
===========================================================================================
###################################################################################################




#####################################################################################################################
########################################## Our Recommender System: Methodology used #################################
#####################################################################################################################
=== We followed these steps in building our recommender model
======= 1. Baselines: We computed global mean, user mean and item means as the baseline predictions
======= 2. We then tried the matrix factorization method using SGD, ALS, CCD, CCD++
======= 3. Further we accounted for user and item biases.
======= 4. Based on output of matrix factorization, we built a feature matrix and then used polynomial ridge regression. 
======= 4. Finally we also implemented the user based collaborative filtering
======= 5. We built an ensemble method of Matrix factorisation with ALS and user based collaborative filtering.
======= 5. We used cross validation to determine the optimal parameters for each method
======= 6. We then compared the cross validation results for different methods 
======= 7. The ensemble method was found to be the best performing model in terms of RMSE.
#####################################################################################################################



#####################################################################################################################
########################################  Contents of zip file and file organisation ################################
#####################################################################################################################
==== The delivered zip file contains the following:
======= 1. README.txt
======= 2. data/ - Contains the data_train.csv and sampleSubmission.csv files
======= 3. run.py - script to generate .csv predictions submitted to Kaggle
======= 4. The other files are as follows:
============== 1. helpers.py            		# Helper functions
============== 2. cross_validation.py   		# Functions to run cross validation
============== 3. ensemble_als_userfilter.py   	# Function to build ensemble model
============== 4. mf_als.py   					# Matrix factorization with ALS
============== 5. mf_ccd.py   					# Matrix factorization with CCD and CCD++
============== 6. mf_sgd.py   					# Matrix factorization with SGD
============== 7. preprocess.py   				# Additional helper functions
============== 8. sim_user.csv   				# User similarity matrix for collaborative filtering (Precomputed)
============== 9. split_data.py   				# Functions to split data into training and test sets
==============10. submit_predictions.py   		# Functions to generate csv submission file
==============11. user_collaborative_filter.py  # User collaborative filter
#####################################################################################################################



