import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
from numpy import savetxt

mlp = MLPClassifier(hidden_layer_sizes=(397,), max_iter=10000,solver='sgd', verbose=True, tol=1e-4, learning_rate_init=0.0001)

#lista_dados_train = np.array(pd.read_csv("mnist_10000.csv", sep=','))
#lista_dados_train = np.array(pd.read_csv("mnist_60000.csv", sep=','))
lista_dados_train = np.array(pd.read_csv("mnist_10000.csv", sep=','))
y_train = lista_dados_train[:,0]
X_train = lista_dados_train[:,1:785]

#lista_dados_test = np.array(pd.read_csv("mnist_100.csv", sep=','))
lista_dados_test = np.array(pd.read_csv("mnist_10000.csv", sep=','))


#mlp.fit(X_train, y_train)
kf = KFold(n_splits=3)
kf.get_n_splits(lista_dados_test)
for train_index, test_index in kf.split(lista_dados_test):
	y_train = lista_dados_test[train_index][:,0]
	X_train = lista_dados_test[train_index][:,1:785]
	y_test = lista_dados_test[test_index][:,0]
	X_test = lista_dados_test[test_index][:,1:785]
	mlp.fit(X_train, y_train)
	print("Current KFolds score: %f" % mlp.score(X_test, y_test))


y_test = lista_dados_test[:,0]
X_test = lista_dados_test[:,1:785]
print("Training set score: %f" % mlp.score(X_train, y_train))

print("Test set score: %f" % mlp.score(X_test, y_test))

saida = mlp.predict(X_test)
pesos = mlp.coefs_
for i in range(len(pesos)):
	savetxt('pesos' + str(i) + '.csv', pesos[i], delimiter=',')
with open('mlp.pkl', 'wb') as f:
    pickle.dump(mlp, f)

print("Rotulo: ", y_test)
print(" Predicao: ", saida)

i = 7
plt.imshow(X_test[i].reshape(28,28), cmap='gray',interpolation='nearest')
print("Rotulo: ", y_test[i])
print("Predicao: ", saida[i])

print(confusion_matrix(y_test, saida))
plt.show();