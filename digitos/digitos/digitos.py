import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from joblib import load

mlp = load("mlp.joblib")
#peso0 = np.array(pd.read_csv("pesos0.csv", sep=','))
#peso1 = np.array(pd.read_csv("pesos1.csv", sep=','))
#peso = np.array([peso0, peso1]).tolist()
#mlp.coefs_ = peso
#mlp.n_outputs_ = 10
#mlp.n_layers_ = 1
def RecognizeDigit(digit):        
    return mlp.predict(digit)

lista_dados_train = np.array(pd.read_csv("mnist_100.csv", sep=','))
y_train = lista_dados_train[:,0]
X_train = lista_dados_train[:,1:785]
for i in range(len(X_train)):
    print("Esperado: ", y_train[i])
    
saida = mlp.predict(X_train)
print(saida)
