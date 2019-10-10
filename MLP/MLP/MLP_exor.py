import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

n = 0.8
eps = 1e-6


nIn  = 2
nOut = 1
camadas = 2         #1 ocultas e 1 de saida
nNeuronios = [2, 1]


nAmostras = 4
d = np.ones(4)
x = np.ones([4, 3])*-1
x[0,0] = 0.
x[0,1] = 0.
x[1,0] = 0.
x[1,1] = 1.
x[2,0] = 1.
x[2,1] = 0.
x[3,0] = 1.
x[3,1] = 1.
d[0] = 0.
d[1] = 1.
d[2] = 1.
d[3] = 0.

def g(x):
	return 1. / (1. + np.exp(-x))

def dg(x):
	_tmp = g(x)
	return _tmp*(1-_tmp)

def Y(_w, _x, nI):
	_l = np.zeros(nI)
	_l = np.dot(_w, _x)
	_y = np.zeros(nI+1)
	for i in range(nI):
		_y[i] = g(_l[i])
	_y[nI] = -1
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

#w = [0, np.ones([nNeuronios[0], nIn+1])/2, np.ones([nOut, nNeuronios[0]+1])/2]
w[1][0] = np.array([.8, .3, -.7])
w[1][1] = np.array([-.6, -.4, .4])
w[2][0] = np.array([.7, -.8, .3])


w0 = w.copy()
saidas = np.zeros(nAmostras)
epocas = 0
continuar = True
E = Eant = 0
Elist = []

while continuar:
	Eant = E
	for amostra in range(nAmostras):
        
        
		#amostra = 3
        
        
		############################## Forward ###
		y[0] = x[amostra]

		l[1], y[1] = Y(w[1], y[0], 2)
		l[2], y[2] = Y(w[2], y[1], 1)
		s = y[2][0]
		saidas[amostra] = s
		
		
		############################## Backward ###
		sigma2 = (d[amostra]-s) * dg(l[2])
		dw2 = n * sigma2 * y[1][:]
		
		
		#de camada 2 para a 1		
		sigma1 = np.zeros(nNeuronios[0])
		for i in range(nNeuronios[0]):
			sigma1[i] = sigma2 * w[2][0,i] * dg(l[1][i])
		
		dw1 = np.zeros([nNeuronios[0], nIn+1])
		for j in range(nNeuronios[0]):
			for i in range(nIn+1):
				dw1[j,i] = n * sigma1[j] * y[0][i]
		
		w[1] = w[1] + dw1
		w[2][0] = w[2][0] + dw2
		
		
	'''for amostra in range(nAmostras):
		y[0] = x[amostra]
		l[1], y[1] = Y(w[1], y[0], len(l[1]))
		l[2], y[2] = Y(w[2], y[1], len(l[2]))
		saidas[amostra] = y[2][1]'''
	
	E = Em(d, saidas, nAmostras)
	Elist.append(E)

	epocas = epocas+1
	continuar = abs(Eant-E)>eps


###############################################################################
############################## Teste #
nTeste = nAmostras
dt = d
xt = x

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
	


print(np.matrix(res.T))

plt.plot(Elist)
plt.ylabel('Elist')
plt.show()
