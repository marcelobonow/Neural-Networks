import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def importData(file, _nIn, _nOut):
	dados = np.array(pd.read_excel(file))
	d = dados[:, _nIn:].copy()
	x = dados[:, :_nIn+1].copy()
	x[:, _nIn:] = np.ones([len(dados),1])*-1
	return x, d, len(dados), _nIn, _nOut

###leituraa dos dados do ex1
#xTreino, dTreino, sTreino, nIn, nOut = importData('373925-Treinamento_projeto_1_MLP.xls', 3, 1)
#xTeste,  dTeste,  sTeste,  nIn, nOut = importData('373922-Teste_projeto_1_MLP.xls', 3, 1)

###leituraa dos dados do ex2
xTreino, dTreino, sTreino, nIn, nOut = importData('373926-Treinamento_projeto_2_MLP.xls', 4, 3)
xTeste,  dTeste,  sTeste,  nIn, nOut  = importData('373923-Teste_projeto_2_MLP.xls', 4, 3)

#x = np.array([[0,0,-1],
#			  [0,1,-1],
#			  [1,0,-1],
#			  [1,1,-1]])
#d = np.array([[1,0,0,1],
#			  [0,1,1,1]]).T
#
#xTreino = xTeste = x
#dTreino = dTeste = d
#sTreino = sTeste = 4

#nIn, nOut = 2, 2

###caracteristicas da rede
aprendizado = 0.1
momentum = 0.9
Emax = 1e-7
epocasMax = 10000		#limite de epocas para o treino

camadas = 2         #1 ocultas e 1 de saida
n = [15, nOut]

def g(x):
	return 1. / (1. + np.exp(-x))	

def dg(x):
	return g(x)*(1.-g(x))


x, d, size = xTreino, dTreino, sTreino
############################################################
############  Treino
w = []		#w = [np.random.random([n[0], nIn+1])*2-1, np.random.random([n[1], n[0]+1])*2-1]
w.append(np.random.random([n[0], nIn+1])*2-1)
for j in range(1, camadas):
	w.append(np.random.random([n[j], n[j-1]+1])*2-1)

l = []		#l = [np.zeros(n[0]),   np.zeros(n[1])]
for j in range(camadas):
	l.append(np.zeros(n[j]))

y = []		#y = [np.zeros(n[0]+1), np.zeros(n[1])]
for j in range(camadas-1):
	y.append(np.zeros(n[j]+1))
y.append(np.zeros(n[camadas-1]))

s = []		#s = [np.ones((n[0])), np.ones((n[1]))]
for j in range(camadas):
	s.append(np.ones((n[j])))

w0 = w.copy()
wn = w.copy()
wa1 = w.copy()
wa2 = w.copy()

epocas = 0
E = 0
Eant = 1
Elist = []

while (abs(Eant-E)>Emax and epocas < epocasMax):
	Eant = E
	E = 0

	for i in range(size):	
		wa2 = wa1.copy()
		wa1 =   w.copy()
		
		############################## Forward ###
		l[0] = np.dot(w[0], x[i])
		for j in range(n[0]): 
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		for c in range(1, camadas):
			l[c] = np.dot(w[c], y[c-1])
			for j in range(n[c]): 
				y[c][j] = g(l[c][j])

		############################## Backward ###
		c = camadas - 1
		for j in range(n[c]):
			s[c][j] = (d[i][j] - y[c][j]) * dg(l[c][j])
			wn[c][j] = w[c][j] + (aprendizado * s[c][j] * y[c-1][j])

		for c in range(camadas-2, 0, -1):
			for j in range(n[c]):
				s[c][j] = np.dot(s[c+1], w[c+1][:, j]) * dg(l[c][j])
				wn[c][j] = w[c][j] + (aprendizado * s[c][j] * y[c-1][j])
		
		for j in range(n[0]):
			s[0][j] = np.dot(s[1], w[1][:, j]) * dg(l[0][j])
			wn[0][j] = w[0][j] + (aprendizado * s[0][j] * x[i])
		
		
		for c in range(camadas):
			w[c] = wn[c] + momentum * (wa1[c] - wa2[c])
	
	for i in range(size):
		############################## Forward ###
		l[0] = np.dot(w[0], x[i])
		for j in range(n[0]): 
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		for c in range(1, camadas):
			l[c] = np.dot(w[c], y[c-1])
			for j in range(n[c]): 
				y[c][j] = g(l[c][j])
			
		er = 0
		for j in range(nOut):
			er = er + ((d[i][j] - y[camadas-1][j])**2)
		E = E + 0.5*er
	E = E/size
	Elist.append(E)
	epocas = epocas +1


############################################################
############  Teste
x, d, size = xTeste,  dTeste,  sTeste

res = np.zeros([size, 2*nOut])
res[:, :nOut] = d
resSat = res.copy()
E = 0

l = []		#l = [np.zeros(n[0]),   np.zeros(n[1])]
for j in range(camadas):
	l.append(np.zeros(n[j]))
y = []		#y = [np.zeros(n[0]+1), np.zeros(n[1])]
for j in range(camadas-1):
	y.append(np.zeros(n[j]+1))
y.append(np.zeros(n[camadas-1]+1))

for i in range(size):
	############################## Forward ###
	l[0] = np.dot(w[0], x[i])
	for j in range(n[0]): 
		y[0][j] = g(l[0][j])
	y[0][n[0]] = -1
	for c in range(1, camadas):
		l[c] = np.dot(w[c], y[c-1])
		for j in range(n[c]): 
			y[c][j] = g(l[c][j])
	for j in range(n[camadas-1]):
		if y[camadas-1][j] < 0.5:
			resSat[i, nOut+j] = 0
		else:
			resSat[i, nOut+j] = 1
		res[i, nOut+j] = y[camadas-1][j]
	er = 0
	for j in range(n[camadas-1]):
		er = er + ((d[i][j] - y[camadas-1][j])**2)
		E = E + 0.5*er 
	E = E + 0.5*er
E = E/size


print(np.matrix(res))

print('Eqm: '+str(E)+'    dEqm: '+str(abs(Elist[len(Elist)-1]-Elist[len(Elist)-2]))+'    epocas: '+str(epocas))

plt.plot(Elist)
plt.ylabel('Elist')
plt.show()
