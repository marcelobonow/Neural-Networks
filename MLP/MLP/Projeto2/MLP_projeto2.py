import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


treinamentoNumero = 0 #Nesse caso, a mudança é no momentum, então ele serve apenas como semente para o random
file = "out" + str(treinamentoNumero) + ".txt"
f = open(file, 'w')
sys.stdout = f #Em vez de jogar para o console, coloca no arquivo de texto, para facilitar para o relatório
print("Versão do compilador: " + sys.version)

np.random.seed(treinamentoNumero) # Mesma seed tanto para com momentum quanto sem momentum, para garantir mesmos pesos

def importData(file, _nIn, _nOut):
	dados = np.array(pd.read_excel(file))
	d = dados[:, _nIn:].copy()
	x = dados[:, :_nIn + 1].copy()
	x[:, _nIn:] = -1
	return x, d, len(dados), _nIn, _nOut

#Dados de treino e de teste
xTreino, dTreino, sTreino, nIn, nOut = importData('373926-Treinamento_projeto_2_MLP.xls', 4, 3)
xTeste,  dTeste,  sTeste,  nIn, nOut = importData('373923-Teste_projeto_2_MLP.xls', 4, 3)

aprendizado = 0.1
momentum = 0 #0 faz o mesmo que não haver momento, para o teste com momento, a variavel recebe 0.9
precisao = 10e-6
#Limite de épocas para o treino, para a precisão atual não é necessário, porém para precisões maiores é essencial
epocasMax = 13000 

#1 oculta e 1 de saída
camadas = 2

n = [15, nOut]

#função de ativação logística
def g(x):
	return 1. / (1. + np.exp(-x))

#derivada de g
def dg(x):
	return g(x) * (1. - g(x))


x, d, size = xTreino, dTreino, sTreino

#Seção de treino
w = []
w.append(np.random.random([n[0], nIn + 1]) * 2 - 1) #inicializa os pesos entre -1 e 1, seed setada pelo numpy
for j in range(1, camadas):

	w.append(np.random.random([n[j], n[j - 1] + 1]) * 2 - 1)#função de ativação logística

# Zera as camadas
l = []
#derivada de g[]
for j in range(camadas):
	l.append(np.zeros(n[j]))

y = []
for j in range(camadas - 1):
	y.append(np.zeros(n[j] + 1))
y.append(np.zeros(n[camadas - 1]))

s = []
for j in range(camadas):
	s.append(np.ones((n[j])))

#Inicia as variaveis auxiliares
w0 = w.copy()
wn = w.copy()
wAnterior1 = w.copy()
wAnterior2 = w.copy()

epocas = 0
erro = 0
erroAnterior = precisao+1 # não precisa ser 1, apenas maior que precisão, para entrar no while
listaErro = [] #Para gerar gráfico

while (abs(erroAnterior - erro) > precisao and epocas < epocasMax):
    erroAnterior = erro
    erro = 0

#Tanto para saber que esta rodando quanto para confirmar o gráfico
    print("Epoca: " + str(epocas) + " Erro: " + str(erroAnterior))

    for i in range(size):
        wAnterior2 = wAnterior1.copy()
        wAnterior1 = w.copy()

        l[0] = np.dot(w[0], x[i])
        for j in range(n[0]):
            y[0][j] = g(l[0][j])
        y[0][n[0]] = -1
        for c in range(1, camadas):
            l[c] = np.dot(w[c], y[c - 1])
            for j in range(n[c]):
                y[c][j] = g(l[c][j])

		#Backward
        c = camadas - 1
        for j in range(n[c]):
            s[c][j] = (d[i][j] - y[c][j]) * dg(l[c][j])
            wn[c][j] = w[c][j] + (aprendizado * s[c][j] * y[c - 1][j])

        for c in range(camadas - 2, 0, -1):
            for j in range(n[c]):
                s[c][j] = np.dot(s[c + 1], w[c + 1][:, j]) * dg(l[c][j])
                wn[c][j] = w[c][j] + (aprendizado * s[c][j] * y[c - 1][j])

        for j in range(n[0]):
            s[0][j] = np.dot(s[1], w[1][:, j]) * dg(l[0][j])
            wn[0][j] = w[0][j] + (aprendizado * s[0][j] * x[i])


        for c in range(camadas):
            w[c] = wn[c] + momentum * (wAnterior1[c] - wAnterior2[c])

    for i in range(size):
        l[0] = np.dot(w[0], x[i])
        for j in range(n[0]):
            y[0][j] = g(l[0][j])
        y[0][n[0]] = -1
        for c in range(1, camadas):
            l[c] = np.dot(w[c], y[c - 1])
            for j in range(n[c]):
                y[c][j] = g(l[c][j])

        er = 0
        for j in range(nOut):
            er = er + ((d[i][j] - y[camadas - 1][j]) ** 2)
        erro = erro + 0.5 * er
    erro = erro / size
    listaErro.append(erro)
    epocas = epocas + 1



#Seção de teste
x, d, size = xTeste,  dTeste,  sTeste

resultado = np.zeros([size, 2 * nOut])
resultado[:, :nOut] = d
resultadoClassificado = resultado.copy()
erro = 0

l = []
for j in range(camadas):
	l.append(np.zeros(n[j]))
y = []
for j in range(camadas - 1):
	y.append(np.zeros(n[j] + 1))
y.append(np.zeros(n[camadas - 1] + 1))

for i in range(size):
	l[0] = np.dot(w[0], x[i])
	for j in range(n[0]):
		y[0][j] = g(l[0][j])
	y[0][n[0]] = -1
	for c in range(1, camadas):
		l[c] = np.dot(w[c], y[c - 1])
		for j in range(n[c]):
			y[c][j] = g(l[c][j])
	for j in range(n[camadas - 1]):
		if y[camadas - 1][j] < 0.5:
			resultadoClassificado[i, nOut + j] = 0
		else:
			resultadoClassificado[i, nOut + j] = 1
		resultado[i, nOut + j] = y[camadas - 1][j]
	er = 0
	for j in range(n[camadas - 1]):
		er = er + ((d[i][j] - resultado[i][j+nOut]) ** 2)
		erro = erro + 0.5 * er #Apesar de na classificação não haver diferença, nos valores em si há
	erro = erro + 0.5 * er
	print("entrada:\t" + str(i) +  "\tesperado:\t" + str(d[i]) + "\tobtido:\t" + str(resultado[i]) + "\n") #Resultado do teste, para a tabela e matriz de confusão
erro = erro / size


print(np.matrix(resultado))

#Erros do teste
print('Eqm: ' + str(erro))
print('dEqm: ' + str(abs(listaErro[len(listaErro) - 1] - listaErro[len(listaErro) - 2]))) #Se tiver sido bloqueado pelo número máximo de épocas
print('epocas: ' + str(epocas))

#Plota num gráfico a lista de erros
plt.plot(listaErro)
plt.ylabel('Elist')
plt.show()
