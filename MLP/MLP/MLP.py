import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

treinamentoNumero = int(input()) #para facilitar os treinos
file = "out" + str(treinamentoNumero) + ".txt" #coloca em um arquivo por teste
f = open(file, 'w')
sys.stdout = f

print("Versão do compilador: " + sys.version)

np.random.seed(treinamentoNumero) #O valor da seed é modificado a cada teste, começando em 0

aprendizado = 0.1
eMax = 10e-6

inputQuantity  = 3
outputQuantity = 1
#1 oculta e 1 de saida
camadas = 2         
n = [10, outputQuantity]


dados = np.array(pd.read_excel('373925-Treinamento_projeto_1_MLP.xls'))
quantidadeAmostras = len(dados)
d = dados[:, 3]
x = np.ones([len(dados), len(dados[0])])
x[:, 3] = x[:, 3]*-1
x[:, :3] = dados[:, :3]


#função de ativação logística
def g(x):
	return 1. / (1. + np.exp(-x))	

#derivada de g
def dg(x):
	return g(x)*(1-g(x))
		
#-----------------------------------------------------
#Seção de treino
w1 = np.random.random([10, 4])*2-1 # inicializa os pesos aleatoriamente entre -1 e 1
w2 = np.random.random(11)*2-1 #inicializa os pesos aleatoriamente entre -1 e 1

l1 = np.zeros(10)
l2 = 0

y1 = np.zeros(11)
y2 = 0
wn1 = w1.copy()
wn2 = w2.copy()

epocas = 0
error = 0
errorAnterior = 1
listaErros = []

s1 = np.ones((10))

while (abs(errorAnterior-error)>eMax and epocas < 3000):
	errorAnterior = error
	error = 0

	for k in range(200):
		l1 = np.dot(w1, x[k])
		for i in range(10): y1[i] = g(l1[i])
		y1[10] = -1
		l2 = np.dot(w2, y1)
		y2 = g(l2)

		#Backward 
		s2 = (d[k] - y2) * dg(l2)
		wn2 = w2 + (aprendizado * s2 * y1)
		
		for i in range(10):
			s1[i] = (s2 * w2[i]) * dg(l1[i])
			wn1[i] = w1[i] + (aprendizado * s1[i] * x[k])

		w1 = wn1
		w2 = wn2

		error = error + 0.5*((d[k] - y2)**2)
	error = error / 200
	listaErros.append(error)
	epocas = epocas +1
print ("Número de épocas: " + str(epocas))
print ("Erro final: " + str(error)) #error é o eqm

#Seção de teste
dadosTeste = np.array(pd.read_excel('373922-Teste_projeto_1_MLP.xls'))
quantidadeTeste = len(dadosTeste)
dt = dadosTeste[:, 3]
xt = np.ones([len(dadosTeste), len(dadosTeste[0])])
xt[:, 0] = xt[:, 0]*-1
xt[:, 1:] = dadosTeste[:, :3]

dt = d[:20]
xt = x[:20,:]

res = np.zeros([2,quantidadeTeste])
res[0,:] = dt
l1t = np.zeros(10)
y1t = np.zeros(11)

error = 0
for k in range(quantidadeTeste):			
    l1t = np.dot(w1, xt[k])
    for i in range(10): y1t[i] = g(l1t[i])
    y1t[10] = -1
    l2t = np.dot(w2, y1t)
    y2t =  g(l2t)
    res[1,k] = y2t
    error = error + 0.5*((dt[k] - y2t)**2)
error = error/200
print("Erro no teste: " + str(error)); #error é o eqm do teste
    

print(np.matrix(res.T))

#Plota num gráfico a lista de erros
plt.plot(listaErros)
plt.title("Resultado do Treino")
plt.ylabel('Erro')
plt.xlabel("Épocas");
plt.show()
