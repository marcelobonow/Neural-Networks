import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###caracteristicas da rede
aprendizado = 0.1
momentum = 0.9
Emax = 1e-6

inputQuantity  = 3
outputQuantity = 3
#1 oculta e 1 de saida
camadas = 2         
n = [15, outputQuantity]

epocasMax = 5000

def importData(file):
	dados = np.array(pd.read_excel(file))
	d = dados[:, 4:].copy()
	x = dados[:, :5].copy()
	x[:, 4:] = np.ones([len(dados),1])*-1
	return x, d, len(dados)

def g(x):
	return 1. / (1. + np.exp(-x))	

def dg(x):
	tmp = g(x)
	return tmp*(1-tmp)


def Treino(w, x, d, size, useMomentum):
	#w = [np.random.random([n[0], inputQuantity+1])*2-1, np.random.random([n[1], n[0]+1])*2-1]
	wn = w.copy()
	wa1 = w.copy()
	wa2 = w.copy()
	l = [np.zeros(n[0]),   np.zeros(n[1])]
	y = [np.zeros(n[0]+1), np.zeros(n[1])]

	epocas = 0
	E = 0
	Eant = 1
	Elist = []

	s = [np.ones((n[0])), np.ones((n[1]))]

	while (abs(Eant-E)>Emax and epocas < epocasMax):
		Eant = E
		E = 0

		for i in range(size):
			wa2 = wa1.copy()
			wa1 = w.copy()
			
			############################## Forward ###
			l[0] = np.dot(w[0], x[i])
			for j in range(n[0]): 
				y[0][j] = g(l[0][j])
			y[0][n[0]] = -1
			l[1] = np.dot(w[1], y[0])
			for j in range(n[1]): 
				y[1][j] = g(l[1][j])

			############################## Backward ###
			for j in range(n[1]):
				s[1][j] = (d[i][j] - y[1][j]) * dg(l[1][j])
				wn[1][j] = w[1][j] + (aprendizado * s[1][j] * y[0][j])

			for j in range(n[0]):
				s[0][j] = np.dot(s[1], w[1][:, j]) * dg(l[0][j])
				wn[0][j] = w[0][j] + (aprendizado * s[0][j] * x[i])

			
			if useMomentum:
				w[0] = wn[0] + momentum * (wa1[0] - wa2[0])
				w[1] = wn[1] + momentum * (wa1[1] - wa2[1])
			else:
				w = wn
		
		for i in range(size):
			############################## Forward ###
			l[0] = np.dot(w[0], x[i])
			for j in range(n[0]): 
				y[0][j] = g(l[0][j])
			y[0][n[0]] = -1
			l[1] = np.dot(w[1], y[0])
			for j in range(n[1]): 
				y[1][j] = g(l[1][j])
				
			er = 0
			for j in range(outputQuantity):
				er = er + ((d[i][j] - y[1][j])**2)
			E = E + 0.5*er
		E = E/size
		Elist.append(E)
		epocas = epocas +1
	return w, Elist, epocas, E


def Teste(w, x, d, size):
	res = np.zeros([size, 6])
	res[:, :3] = d
	E = 0
	l = [np.zeros(n[0]),   np.zeros(n[1])]
	y = [np.zeros(n[0]+1), np.zeros(n[1])]
	for i in range(size):
		############################## Forward ###
		l[0] = np.dot(w[0], x[i])
		for j in range(n[0]): 
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		l[1] = np.dot(w[1], y[0])
		for j in range(n[1]): 
			y[1][j] = g(l[1][j])
		res[i, 3:] = y[1]
		er = 0
		for j in range(n[1]):
			er = er + ((d[i][j] - y[1][j])**2)
			E = E + 0.5*er 
		E = E + 0.5*er
	resSat = res.copy()
	for i in range(size):		#saturacao
		if res[i, 3] > res[i, 4] and res[i, 3] > res[i, 5]:
			resSat[i, 3:] = [1, 0, 0]
		elif res[i, 4] > res[i, 5]:
			resSat[i, 3:] = [0, 1, 0]
		else:
			resSat[i, 3:] = [0, 0, 1]
		
	acertos = np.zeros(3)
	for i in range(size):		#teste acercos
		if (resSat[i, 3] == d[i, 0]): acertos = acertos + [1, 0, 0]
		if (resSat[i, 4] == d[i, 1]): acertos = acertos + [0, 1, 0]
		if (resSat[i, 5] == d[i, 2]): acertos = acertos + [0, 0, 1]
		#if (resSat[i, 3] and d[i, 0]) or (resSat[i, 4] and d[i, 1]) or (resSat[i, 5] and d[i, 2]):
		#	acertos = acertos + 1
	return res, resSat, E/size, acertos

###leituraa dos dados
xTreino, dTreino, sTreino = importData('373926-Treinamento_projeto_2_MLP.xls')
xTeste,  dTeste,  sTeste  = importData('373923-Teste_projeto_2_MLP.xls')

w0 = []
wf = []
wfm = []
Elist = []
Elistm = []
epocas = []
epocasm = []
E = []
Em = []
res = []
resm = []
resSat = []
resSatm = []
err = []
errm = []
acertos = []
acertosm = []

nTreinos = 5

for i in range(nTreinos):
	w = [np.random.random([n[0], inputQuantity+1])*2-1, np.random.random([n[1], n[0]+1])*2-1]
	_w0 = w.copy()
	
	_w,  _Elist,  _epocas,  _E  = Treino(_w0, xTreino, dTreino, sTreino, False)
	_wm, _Elistm, _epocasm, _Em = Treino(_w0, xTreino, dTreino, sTreino, True)
	_res,  _resSat,  _err,  _acertos  = Teste(_w, xTeste,  dTeste,  sTeste)
	_resm, _resSatm, _errm, _acertosm = Teste(_w, xTeste,  dTeste,  sTeste)

	w0.append(_w0)
	wf.append(_w)
	Elist.append(_Elist)
	epocas.append(_epocas)
	E.append(_E)
	
	res.append(_res)
	resSat.append(_resSat)
	err.append(_err)
	acertos.append(_acertos)
	
	wfm.append(_wm)
	Elistm.append(_Elistm)
	epocasm.append(_epocasm)
	Em.append(_Em)
	
	resm.append(_resm)
	resSatm.append(_resSatm)
	errm.append(_errm)
	acertosm.append(_acertosm)

#print(np.matrix(res.T))

for i in range(nTreinos):
	print()
	print('--------------------------------------------------------------------------------')
	print()
	print('Treino '+str(i))
	print('Eqm: '+str(E[i]) +'    epocas: '+str(epocas[i]) +'   sem momentum')
	print('Eqm: '+str(Em[i])+'    epocas: '+str(epocasm[i])+'   momentum 0.9')
	
	plt.plot(Elist[i])
	plt.plot(Elistm[i], 'g')
	plt.ylabel('Elist['+str(i)+']')
	plt.legend(['sem momentum', 'momentum = 0.9'])
	plt.show()
	
	aux = np.ones((18,13))
	aux[:,  :4 ] = xTeste[:, :4]
	aux[:, 4:7 ] = dTeste
	aux[:, 7:10] = resSat [i][:, 3:6]
	aux[:, 10: ] = resSatm[i][:, 3:6]
	
	print('x1 x2 x3 x4 D1 D2 D3 Y1 Y2 Y3 Y1m Y2m Y3m                     #Y s/momentum   Ym c/momentum')
	print(np.matrix(aux))
	print()
	print('acertos sem momentum:  '+str(acertos [i]))
	print('acertos com momentum:  '+str(acertosm[i]))
	
	print()
	print('W0 inicial:')
	print(w0[i][0])
	print('W0 inicial:')
	print(w0[i][1])
	
	print()
	print('W0 sem momentum:')
	print(wf[i][0])
	print('W1 sem momentum:')
	print(wf[i][1])
	
	print()
	print('W0 com momentum:')
	print(wfm[i][0])
	print('W1 com momentum:')
	print(wfm[i][1])

































