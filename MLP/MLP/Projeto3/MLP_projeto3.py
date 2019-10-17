import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Network:

    def __init__(self,entry, n1, learnRate, momentum, precision, maxEra=1000):
        self.learnRate = learnRate
        self.momentum = momentum
        self.precision = precision
        self.maxEra = maxEra
        self.layers = 2 #Uma de entrada e uma de saida
        self.n1 = n1
        self.entry = entry

    def train(self):
        self.era = 0
        self.em = 0
        


file = "out.txt"
f = open(file, 'w')
sys.stdout = f #Em vez de jogar para o console, coloca no arquivo de texto, para facilitar
               #para o relatório
print("Versão do compilador: " + sys.version)

seed = 15000
np.random.seed(seed)

#Dados de treino e de teste
dTrain, trainQuantity = importData('373928-Treinamento_projeto_3_MLP.xls')
dTest, testQuantity = importData('373924-Teste_projeto_3_MLP.xls')

firstNetwork = Network(5, 10, 0.1, 0, 10e-6, 13000)


n = [15, nOut]

def importTrainData(file):
        d = np.array(pd.read_excel(file))
        return d, len(d)

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
erroAnterior = precisao + 1 # não precisa ser 1, apenas maior que precisão, para entrar no while
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
        er = er + ((d[i][j] - resultado[i][j + nOut]) ** 2)
        erro = erro + 0.5 * er #Apesar de na classificação não haver diferença, nos valores em si há
    erro = erro + 0.5 * er
    print("entrada:\t" + str(i) + "\tesperado:\t" + str(d[i]) + "\tobtido:\t" + str(resultado[i]) + "\n") #Resultado do teste, para a tabela e matriz de confusão
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
