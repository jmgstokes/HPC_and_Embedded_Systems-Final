#timing training of a random forest model on 4 cores
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
def random_forest(samples,features,trees,jobs):
	# define dataset
	X, y = make_classification(n_samples=samples, n_features=features, n_informative=15, n_redundant=5, random_state=3)
	# define the model
	model = RandomForestClassifier(n_estimators=trees, n_jobs=jobs)
	# record current time
	start = time()
	# fit the model
	model.fit(X, y)
	# record current time
	end = time()
	# report execution time
	result = end - start
	return result