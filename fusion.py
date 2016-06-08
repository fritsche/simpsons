
import numpy as np

from feature_extraction import extraction
from classification import classifier

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix 


if __name__=='__main__':

	fusion = 'sum'

	# features_list = ['color_histogram']
	# classifiers = ['svm']

	features_list = ['color_histogram', 'hu_moments', 'histogram_of_oriented_gradients']
	classifiers = ['svm', 'lda', 'knn']

	preds = []
	pred_probs = []

	for descriptor in features_list :

		X_train = []
		y_train = []
		X_test = []
		y_test = []

		X_train, y_train = extraction ("data/Train/", descriptor)
		X_test, y_test = extraction ("data/Valid/", descriptor)

		scaler = preprocessing.MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

		X_train = np.asarray(X_train)
		X_test = np.asarray(X_test)

		for method in classifiers:

			clf = classifier (method, X_train, y_train)

			y_pred = clf.predict(X_test) 
			y_predProb = clf.predict_proba(X_test) 

			preds.append(y_pred)
			pred_probs.append(y_predProb)

	if fusion == 'sum' :
		matrix = np.zeros(pred_probs[0].shape)
		for pred_prob in pred_probs:
		 	matrix = matrix + pred_prob
		y_pred = np.argmax(matrix, axis=1)

	size = len(y_test)
	score = 0
	for x in xrange(0, size):
		score += (y_test[x] == y_pred[x])
	print ("Score:", float(score)/float(size))

	cm = confusion_matrix(y_test, y_pred)
	print cm
