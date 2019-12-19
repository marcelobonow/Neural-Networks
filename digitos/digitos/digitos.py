import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(1500, 1200), max_iter=10000,solver='sgd', verbose=True, tol=1e-4, learning_rate_init=0.0001)
peso0 = np.array(pd.read_csv("pesos0.csv", sep=','))
peso1 = np.array(pd.read_csv("pesos1.csv", sep=','))
peso = np.stack([peso0, peso1])
mlp.coefs_ = peso

def RecognizeDigit(digit):        
    return mlp.predict(digit)

lista_dados_train = np.array(pd.read_csv("mnist_100.csv", sep=','))
y_train = lista_dados_train[:,0]
X_train = lista_dados_train[:,1:785]
for i in range(len(X_train)):
    print("Esperado: ", y_train[i])
    print("Previsto: ", RecognizeDigit(X_train[i]))

