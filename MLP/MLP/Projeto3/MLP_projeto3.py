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
        self.hiddenSize = n1
        self.inputSize = entry
        self.outputSize = 1
        self.x = np.zeros(entry)
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize) # da entrada variavel até a cama escondida
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) # da camada escondida até a saida

    def train(self):
        self.era = 0
        self.em = 0
    
    def feedFoward(self, x):
        self.y1 = sigmoid(np.dot(x, self.w1))
        self.y2 = sigmoid(np.dot(self.y1, self.w2)) #produto da camada escondida para a saida
        return self.y2

    def backward(self, x, y, output):
        self.error = y - output
        self.delta = self.error * sigmoid(output)
        self.y1_error = self.delta.dot(self.w2.T)
        self.y1_delta = self.y1_error * dSigmoid(self.y1)
        self.w1 += x.T.dot(self.y1_delta) #Ajusta de input para escondida
        self.w2 += self.y1.T.dot(self.delta) #Ajusta de escondida para a saída

    def shiftInput(self, newResult):
        for i in range(1, self.inputSize):
            self.x[i - 1] = self.x[i]
        self.x[self.inputSize - 1] = newResult

    def train(self, x, y):
        output = self.feedFoward(x)
        self.backward(x,y,output)
        shiftInput(output)

def importData(file):
    d = np.array(pd.read_excel(file))
    return d, len(d)

#função de ativação logística
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#derivada de g
def dSigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))

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

for i in range(30):
    firstNetwork.train(firstNetwork.x, dTrain)

epocas = 0
erro = 0
erroAnterior = precisao + 1 # não precisa ser 1, apenas maior que precisão, para entrar no while
listaErro = [] #Para gerar gráfico

##Erros do teste
#print('Eqm: ' + str(erro))
#print('dEqm: ' + str(abs(listaErro[len(listaErro) - 1] - listaErro[len(listaErro) - 2]))) #Se tiver sido bloqueado pelo número máximo de épocas
#print('epocas: ' + str(epocas))

##Plota num gráfico a lista de erros
#plt.plot(listaErro)
#plt.ylabel('Elist')
#plt.show()
