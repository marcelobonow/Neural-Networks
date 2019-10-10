import pandas as pd
import numpy as np
#import math
import matplotlib.pyplot as plt

n = 0.1
eps = 1e-6


nIn  = 3
nOut = 1
camadas = 2         #1 ocultas e 1 de saida
nNeuronios = [10, nOut]



dados = np.array(pd.read_excel('373925-Treinamento_projeto_1_MLP.xls'))
nAmostras = len(dados)
d = dados[:, 3]
x = np.ones([len(dados), len(dados[0])])
x[:, 0] = x[:, 0]*-1
x[:, 1:4] = dados[:, :3]



def g(x):
	return 1. / (1. + np.exp(-x))
	#return math.tanh(x)

def dg(x):
	return g(x)*(1-g(x))
	#return 1-(math.tanh(x)**2)

def Y(_w, _x, nI):
	_l = np.zeros(nI)
	_l = np.dot(_w, _x)
	_y = np.zeros(nI+1)
	_y[0] = -1
	for i in range(len(_l)):
		_y[i+1] = g(_l[i])
	return _l, _y

def Em(_d, _y, _p):
	res = 0
	for i in range(_p):
		res = res + 0.5*((_d[i]-_y[i])**2)
	return res/_p

############################### treino #
w = [0, np.random.random([nNeuronios[0], nIn+1])*2-1, np.random.random([nOut, nNeuronios[0]+1])*2-1]
l = [0,               np.zeros(nNeuronios[0]),   np.zeros(nNeuronios[1])]
y = [np.zeros(nIn+1), np.zeros(nNeuronios[0]+1), np.zeros(nNeuronios[1]+1)]


w = [0, 
	 np.array([[ 6.79187995e-01, -5.13015456e-01, -1.32749444e-01, 9.56402741e-01],
		[-4.43741070e-01, -5.36158120e-01, -8.76989330e-01,
		  1.02935980e-01],
		[-9.68866197e-01, -8.51903155e-01, -8.75329113e-01,
		 -9.18558625e-01],
		[-6.98148285e-01,  2.27233051e-01,  5.35159598e-01,
		 -2.10754033e-01],
		[-7.15788753e-01,  1.04130252e-01, -8.54390866e-01,
		 -5.26829315e-02],
		[ 5.14693188e-01,  8.07540584e-01, -9.23034503e-01,
		  5.47426165e-01],
		[ 3.99539513e-01,  5.39859110e-01, -2.85016027e-01,
		  9.85265915e-01],
		[-6.60281553e-01, -5.88052569e-01,  5.68015104e-01,
		 -4.55039472e-01],
		[ 6.94302155e-01,  5.78379096e-01, -5.51528374e-01,
		 -7.22510774e-01],
		[ 2.16244125e-01, -3.90167623e-01, -8.38588366e-04,
		  8.79687807e-01]]), 
	 np.array([[ 0.70093413,  0.18212131,  0.03674828, -0.29695673, -0.43768653,
		 -0.57053749,  0.75639604,  0.71644495, -0.42575612,  0.34764752,
		  0.98793837]])]
#w = [0, np.ones([nNeuronios[0], nIn+1]), np.ones([nOut, nNeuronios[0]+1])]


w0 = w.copy()
saidas = np.zeros(nAmostras)
epocas = 0
continuar = True
E = Eant = 0
Elist = []

while continuar:
	Eant = E
	for amostra in range(nAmostras):
		############################## Forward ###
		y[0] = x[amostra]

		l[1], y[1] = Y(w[1], y[0], len(l[1]))
		l[2], y[2] = Y(w[2], y[1], len(l[2]))
		saidas[amostra] = y[2][1]

		############################## Backward ###
		sigma2 = (d[amostra]-y[2][1]) * dg(l[2])
		dw2 = n * sigma2 * y[1][:]
		
		#de camada 2 para a 1		
		sigma1 = np.zeros(nNeuronios[0])
		for i in range(nNeuronios[0]):
			sigma1[i] = -sigma2 * w[2][0,i+1] * dg(l[1][i])
		
		dw1 = np.zeros([nNeuronios[0], nIn+1])
		for i in range(nNeuronios[0]):
			dw1[i,:] = n * sigma1[i] * y[0][:]
		
		
		w[1]      = w[1]      + dw1
		w[2][0,:] = w[2][0,:] + dw2
		
	for amostra in range(nAmostras):
		y[0] = x[amostra]
		l[1], y[1] = Y(w[1], y[0], len(l[1]))
		l[2], y[2] = Y(w[2], y[1], len(l[2]))
		saidas[amostra] = y[2][1]
	
	E = Em(d, saidas, nAmostras)
	Elist.append(E)

	epocas = epocas+1
	continuar = abs(Eant-E)>eps


###############################################################################
############################## Teste #

dadosTeste = np.array(pd.read_excel('373922-Teste_projeto_1_MLP.xls'))
nTeste = len(dadosTeste)
dt = dadosTeste[:, 3]
xt = np.ones([len(dadosTeste), len(dadosTeste[0])])
xt[:, 0] = xt[:, 0]*-1
xt[:, 1:] = dadosTeste[:, :3]

lt = [0,               np.zeros(nNeuronios[0]),   np.zeros(nNeuronios[1])]
yt = [np.zeros(nIn+1), np.zeros(nNeuronios[0]+1), np.zeros(nNeuronios[1]+1)]

#res = [dt, np.zeros(nTeste)]
res = np.zeros([2,nTeste])
res[0,:] = dt

for amostra in range(nTeste):
	yt[0] = x[amostra]

	lt[1], yt[1] = Y(w[1], yt[0], len(lt[1]))
	lt[2], yt[2] = Y(w[2], yt[1], len(lt[2]))
	
	res[1,amostra] = yt[2][1]
	


print(np.matrix(res.T*10))

plt.plot(Elist)
plt.ylabel('Elist')
plt.show()
