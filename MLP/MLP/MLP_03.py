import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def importData(file):
	x = np.array(pd.read_excel(file))
	return x, len(x)

#Le dados de treino e teste
xTreino, sTreino = importData('373928-Treinamento_projeto_3_MLP.xls')
xTeste,  sTeste  = importData('373924-Teste_projeto_3_MLP.xls')

#caracteristicas da rede, dado pelo projeto
aprendizado = 0.1
momentum = 0.8
Emax = 0.5e-6
epocasMax = 1000		#limite de epocas para o treino

inputQuantity, nOut =  15, 1
#1 ocultas e 1 de saida
camadas = 2         
n = [25, nOut]

def g(x):
	return 1. / (1. + np.exp(-x))

def dg(x):
	return g(x)*(1.-g(x))


#-----------------------------------------------
#Seção de Treino
size = sTreino
x = np.zeros([size+inputQuantity+1, 1])
x[:size] = xTreino
x[size:] = xTreino[:inputQuantity+1]

w = [np.random.random([n[0], inputQuantity+1])*2-1, np.random.random([n[1], n[0]+1])*2-1]
w0 = w.copy()
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
		wa1 =   w.copy()
		
		xt = x[i:i+inputQuantity+1].T.copy()
		xt[0, inputQuantity] = -1
		d = x[i+inputQuantity]

		############################## Forward ###
		l[0] = np.dot(w[0], xt.T)
		for j in range(n[0]):
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		l[1] = np.dot(w[1], y[0])
		for j in range(n[1]): 
			y[1][j] = g(l[1][j])

		############################## Backward ###
		j = 0
		s[1][j] = (d - y[1][j]) * dg(l[1][j])
		wn[1][j] = w[1][j] + (aprendizado * s[1][j] * y[0][j])

		for j in range(n[0]):
			s[0][j] = np.dot(s[1], w[1][:, j]) * dg(l[0][j])
			wn[0][j] = w[0][j] + (aprendizado * s[0][j] * x[i])
		
		w[0] = wn[0] + momentum * (wa1[0] - wa2[0])
		w[1] = wn[1] + momentum * (wa1[1] - wa2[1])
		#w = wn
	
	for i in range(size):
		xt = x[i:i+inputQuantity+1].T.copy()
		xt[0, inputQuantity] = -1
		d = x[i+inputQuantity]

		############################## Forward ###
		l[0] = np.dot(w[0], xt.T)
		for j in range(n[0]): 
			y[0][j] = g(l[0][j])
		y[0][n[0]] = -1
		
		l[1] = np.dot(w[1], y[0])
		for j in range(n[1]): 
			y[1][j] = g(l[1][j])
			
		er = 0
		j=0	#for j in range(nOut):
		er = er + ((d - y[1][j])**2)
		E = E + 0.5*er
	E = E/size
	Elist.append(E)
	epocas = epocas +1


############################################################
############  Teste
size =  sTeste
x = np.zeros([size+inputQuantity+1, 1])
x[:size] = xTeste
x[size:] = xTeste[:inputQuantity+1]

res = np.zeros([size, 2])
#res[:, :nOut] = d
E = 0
eam = 0
l = [np.zeros(n[0]),   np.zeros(n[1])]
y = [np.zeros(n[0]+1), np.zeros(n[1])]
for i in range(size):
	xt = x[i:i+inputQuantity+1].T.copy()
	xt[0, inputQuantity] = -1
	d = x[i+inputQuantity]

	res[i, 0] = d

	############################## Forward ###
	l[0] = np.dot(w[0], xt.T)
	for j in range(n[0]): 
		y[0][j] = g(l[0][j])
	y[0][n[0]] = -1
	
	l[1] = np.dot(w[1], y[0])
	for j in range(n[1]):
		y[1][j] = g(l[1][j])
	res[i, 1] = y[1][0]
	
	eam = eam + abs(res[i, 0] - res[i, 1])
	
	j = 0
	er = ((d - y[1][j])**2)
	E = E + 0.5*er
E = E/size
eam = eam/size


print(np.matrix(res))

print('Eqm: '+str(E)+'       Eam: '+str(eam)+'    epocas: '+str(epocas))
print('dEqm: '+str(abs(Elist[len(Elist)-1]-Elist[len(Elist)-2])))

plt.plot(Elist)
plt.ylabel('Elist')
plt.show()
