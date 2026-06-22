import numpy as np
import sklearn

# You are allowed to import any submodules of numpy or sklearn e.g. sklearn.metrics.accuracy_score to calculate accuracy of a learnt model
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_map, my_params etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	#Handle single challenge passed as 1D array
	if X.ndim == 1:
		X = X[np.newaxis, :]

	#Convert each challenge bit to +/-1, then build pairwise products
	#between even-indexed and odd-indexed positions.

	n        = X.shape[0]
	z        = 1.0 - 2.0 * X.astype( np.float64 )
	z_e      = z[:, 0::2]
	z_o      = z[:, 1::2]
	pairs    = ( z_e[:, :, np.newaxis] * z_o[:, np.newaxis, :] ).reshape( n, 256 )
	X_map    = np.hstack( [z, pairs] )

	return X_map

################################
# Non Editable Region Starting #
################################
def my_params( X_map, X_raw, y ):
################################
#  Non Editable Region Ending  #
################################


	my_params = {
		'C'       : 0.9,
		'tol'     : 1e-4,
		'max_iter': 5000,
		'dual'    : False,
	}

	return my_params
