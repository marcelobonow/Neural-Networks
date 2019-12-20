import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import load

mlp = load("mlp.joblib")
def RecognizeDigit(digit):        
    return mlp.predict(digit)

lista_dados_train = np.array(pd.read_csv("mnist_100.csv", sep=','))
y_train = lista_dados_train[:,0]
X_train = lista_dados_train[:,1:785]
for i in range(len(X_train)):
    print("Esperado: \t", y_train[i])
    print("Resultado: \t", RecognizeDigit(X_train[i].reshape(1,784)))
    print("\n")