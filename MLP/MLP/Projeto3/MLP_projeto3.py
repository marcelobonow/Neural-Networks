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
        self.x = np.zeros(entry+1)
        self.x[entry] = -1
        self.lastEqm = 0
        self.eqm = 1
        self.era = 0
        self.errorList = []
        self.y1 = np.zeros(self.hiddenSize+1)
        self.y2 = np.zeros(self.outputSize)
        self.w1 = np.random.randn(self.inputSize+1, self.hiddenSize+1) # da entrada variavel até a cama escondida
        self.w2 = np.random.randn(self.hiddenSize+1, self.outputSize) # da camada escondida até a saida
    
    def feedFoward(self, y):
        self.y1 = sigmoid(np.dot(self.x, self.w1)) #Entrada para escondida
        self.y1[len(self.y1)-1] = -1
        self.y2 = sigmoid(np.dot(self.y1, self.w2)) # produto da camada escondida para a saida
        results = self.y2
        self.backward(self.x, y, self.y2)
        self.shiftInput(self.y2[i])
        return results

    def backward(self, x, y, output):
        self.error = y - output
        self.delta = self.error * dSigmoid(output)
        self.y1_error = self.delta * self.w2
        self.y1_delta = self.y1_error * dSigmoid(self.y1)
        self.w1 += x.T.dot(self.y1_delta) #Ajusta de input para escondida
        self.w2 += self.y1.T.dot(self.delta) #Ajusta de escondida para a saída

    def shiftInput(self, newResult):
        for i in range(1, self.inputSize):
            self.x[i - 1] = self.x[i]
        self.x[self.inputSize - 1] = newResult
        self.x[self.inputSize] = -1

    def train(self, y):
        self.x = np.zeros(self.inputSize+1)
        self.x[self.inputSize] = -1
        era = 0
        while (abs(self.lastEqm - self.eqm) > self.precision and self.era < self.maxEra):
            self.lastEqm = self.eqm
            erro = 0
            for i in range(len(y)):
                output = self.feedFoward(y[i])
                erro = erro + 0.5 * ((y[i] - output) ** 2)
                self.backward(self.x, y, output)
                self.shiftInput(output)
            
            self.eqm = erro / len(y)
            self.errorList.append(self.eqm)
            print("erro na era: " + str(self.era) + "  " + str(self.eqm))
            self.era += 1
        print("Fim do treino com erro: " + str(self.eqm))

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

seed = 500
np.random.seed(seed)

#Dados de treino e de teste
dTrain, trainQuantity = importData('373928-Treinamento_projeto_3_MLP.xls')
dTest, testQuantity = importData('373924-Teste_projeto_3_MLP.xls')

firstNetwork = Network(trainQuantity, 5, 10, 0.4, 0, 0.5e-6, 1000)
secondNetwork = Network(trainQuantity, 10, 15, 0.4, 0, 0.5e-6, 1000)
thirdNetwork = Network(trainQuantity, 15, 25, 0.01, 0, 0.5e-6, 1000)


firstNetwork.train(dTrain)
secondNetwork.train(dTrain)
thirdNetwork.train(dTrain)

print("Rede um teve erro: " + str(firstNetwork.eqm) + "em: " + str(firstNetwork.era))
print("Rede dois teve erro: " + str(secondNetwork.eqm) + "em: " + str(secondNetwork.era))
print("Rede tres teve erro: " + str(thirdNetwork.eqm) + "em: " + str(thirdNetwork.era))

##Erros do teste
#print('Eqm: ' + str(erro))
#print('dEqm: ' + str(abs(listaErro[len(listaErro) - 1] -
#listaErro[len(listaErro) - 2]))) #Se tiver sido bloqueado pelo número máximo
#de épocas
#print('epocas: ' + str(epocas))

##Plota num gráfico a lista de erros
plt.plot(firstNetwork.errorList)
plt.plot(secondNetwork.errorList)
plt.plot(thirdNetwork.errorList)
plt.ylabel('Error')
plt.show()
