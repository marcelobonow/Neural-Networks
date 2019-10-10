import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn

from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier


aprendizado = 0.1
eps = 1e-6

nIn  = 3
nOut = 1
camadas = 2         #1 ocultas e 1 de saida
n = [10, nOut]

dados = np.array(pd.read_excel('373925-Treinamento_projeto_1_MLP.xls'))
nAmostras = len(dados)
d = dados[:, 3]
x = np.ones([len(dados), len(dados[0])])
x[:, 3] = x[:, 3]*-1
x[:, :3] = dados[:, :3]



def g(x):
	return 1. / (1. + np.exp(-x))	

def dg(x):
	return g(x)*(1-g(x))

###############################################################################
############################### treino #


mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-6, random_state=1,
                    learning_rate_init=.1,
					activation='logistic')




###############################################################################
############################## Teste #

dadosTeste = np.array(pd.read_excel('373922-Teste_projeto_1_MLP.xls'))
nTeste = len(dadosTeste)
dt = dadosTeste[:, 3]
xt = np.ones([len(dadosTeste), len(dadosTeste[0])])
xt[:, 0] = xt[:, 0]*-1
xt[:, 1:] = dadosTeste[:, :3]