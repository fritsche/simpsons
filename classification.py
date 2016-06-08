
import sys

import numpy as np
from sklearn.lda import LDA
from sklearn.metrics import confusion_matrix 
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV

def GridSearch(X_train, y_train):

        # define range dos parametros
        C_range = 2. ** np.arange(-5,15,2)
        gamma_range = 2. ** np.arange(3,-15,-2)
        k = [ 'rbf']
        #k = ['linear', 'rbf']
        param_grid = dict(gamma=gamma_range, C=C_range, kernel=k)

        # instancia o classificador, gerando probabilidades
        srv = svm.SVC(probability=True)

        # faz a busca
        grid = GridSearchCV(srv, param_grid, n_jobs=-1, verbose=True)
        grid.fit (X_train, y_train)

        # recupera o melhor modelo
        model = grid.best_estimator_

        # imprime os parametros desse modelo
        print grid.best_params_
        return model

def print_usage ():
	print("usage:")
	print("\t python classification.py path/training_base_file path/test_base_file method")
	print("\t\t<training_base_file>: text file with the training base" )
	print("\t\t<test_base_file>: text file with the test base" )
	print("\t\t<method>: classification method" )

def classifier (method, X_train, y_train):
	if method == "lda" :
		clf = LDA() 
	elif method == "knn" :
		clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
	elif method == "svm" :
		clf = GridSearch(X_train, y_train)
	else:
		print ("Unknown classifier method ", method)

	clf.fit(X_train, y_train)

	return clf

if __name__=='__main__':

	if len(sys.argv) == 4 : # train_file test_file method
		train_file = sys.argv[1]
		test_file = sys.argv[2]
		method = sys.argv[3]
	else :
		print ("Bad usage");
		print ("The correct usage is: ");
		print_usage()
		sys.exit(1)

	X_train = []
	y_train = []
	X_test = []
	y_test = []

	with open(train_file) as f:
	    content = f.readlines()

	lines, features = content[0].split()
	for x in range(1,int(lines)+1):
		data = content[x].split()
		data = map (float, data)
		X_train.append(data[:-1])
		y_train.append(data[-1])

	with open(test_file) as f:
	    content = f.readlines()

	lines, features = content[0].split()
	for x in range(1,int(lines)+1):
		data = content[x].split()
		data = map (float, data)
		X_test.append(data[:-1])
		y_test.append(data[-1])

	scaler = preprocessing.MinMaxScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	X_train = np.asarray(X_train)
	X_test = np.asarray(X_test)

	clf = classifier (method, X_train, y_train)

	y_pred = clf.predict(X_test) 
	y_predProb = clf.predict_proba(X_test) 

	# mostra o resultado do classificador na base de teste
	print "Score", clf.score(X_test, y_test)

	# cria a matriz de confusao
	cm = confusion_matrix(y_test, y_pred)
	print cm

	print "y_pred"
	print y_pred

	print "y_predProb"
	print y_predProb

