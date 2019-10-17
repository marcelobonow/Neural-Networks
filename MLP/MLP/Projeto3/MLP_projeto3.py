import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Network:

    def __init__(self, trainLength, entry, n1, learnRate, momentum, precision, maxEra=1000):
        self.learnRate = learnRate
        self.momentum = momentum
        self.precision = precision
        self.maxEra = maxEra
        self.hiddenSize = n1
        self.inputSize = entry
        self.outputSize = 1
        self.x = np.zeros(entry)
        self.lastEqm = 1
        self.eqm = 0
        self.era = 0
        self.errorList = []
        self.allX = np.zeros(shape=(trainLength, entry))
        self.y1 = np.zeros(shape=(trainLength, self.hiddenSize))
        self.y2 = np.zeros(shape=(trainLength, self.outputSize))
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize) # da entrada variavel até a cama escondida
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize) # da camada escondida até a saida
    
    def feedFoward(self, y):
        results = np.zeros(shape=(len(y),1))
        for i in range(len(y)):
            self.y1[i] = sigmoid(np.dot(self.x, self.w1)) #Entrada para escondida
            self.y2[i] = sigmoid(np.dot(self.y1[i], self.w2)) # produto da camada escondida para a saida
            results[i] = self.y2[i]
            self.allX[i] = self.x
            self.shiftInput(self.y2[i])
        return results

    def backward(self, x, y, output):
        self.error = y - output
        self.delta = self.error * dSigmoid(output)
        self.y1_error = self.delta.dot(self.w2.T)
        self.y1_delta = self.y1_error * dSigmoid(self.y1)
        self.w1 += x.T.dot(self.y1_delta) #Ajusta de input para escondida
        self.w2 += self.y1.T.dot(self.delta) #Ajusta de escondida para a saída

    def shiftInput(self, newResult):
        for i in range(1, self.inputSize):
            self.x[i - 1] = self.x[i]
        self.x[self.inputSize - 1] = newResult

    def train(self, y):
        self.lastEqm = self.eqm
        output = self.feedFoward(y)
        self.backward(self.allX,y,output)
        erro = 0
        for i in range(len(y)):
            erro = erro + 0.5 * ((y[i] - output[i]) ** 2)
        self.eqm = erro / len(y)
        self.errorList.append(self.eqm)
        print("erro na era: " + str(self.era) + "  " + str(self.eqm))
        self.era += 1

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

firstNetwork = Network(trainQuantity, 5, 10, 0.1, 0, 10e-6, 13000)

while (abs(firstNetwork.lastEqm - firstNetwork.eqm) > firstNetwork.precision and firstNetwork.era < firstNetwork.maxEra):
    firstNetwork.train(dTrain)


##Erros do teste
#print('Eqm: ' + str(erro))
#print('dEqm: ' + str(abs(listaErro[len(listaErro) - 1] -
#listaErro[len(listaErro) - 2]))) #Se tiver sido bloqueado pelo número máximo
#de épocas
#print('epocas: ' + str(epocas))

##Plota num gráfico a lista de erros
plt.plot(firstNetwork.errorList)
plt.ylabel('Error')
plt.show()
