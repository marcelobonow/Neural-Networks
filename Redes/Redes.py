import sys
import pandas as pd
import numpy as np
import random as rng
from sklearn.model_selection import train_test_split

f = open('out.txt', 'w')
sys.stdout = f
print("Versão do compilador: " + sys.version)

def activation(value):
    if value < 0.0:
        return -1
    else:
        return 1
def GetClass(value):
    if value == - 1:
        return "P1"
    else:
        return "P2"
def SquareMeanError(data, w, label):
    dataQuantity = len(data)
    eqm = 0
    for i in range(dataQuantity):
        xb = np.hstack((bias, data.iloc[i].values))
        u = np.dot(w[0], xb)
        y = label.iloc[i]

        eqm = eqm + (y - u)**2
    eqm = eqm/dataQuantity
    return eqm

data = pd.read_csv('dados.csv', sep=';', decimal = ',')
maxEpocas = 10000

values = data[data.columns[0:4]]
label = data[data.columns[4]]

x_train, x_test, y_train, y_test = train_test_split(values, label, test_size = .3, random_state = 14)
trainQuantity = len(x_train)
testQuantity = len(x_test)

print("Iniciando treinamento com " + str(trainQuantity) + " amostras \n")
print("Entradas de treino \n" + x_train.to_string())
print("Labels de treino \n" + y_train.to_string())
print("Entradas de teste \n" + x_test.to_string())
print("Labels de teste \n" + y_test.to_string())

bias = -1
learnRate = 0.0025
precision = 0.000001
w = np.zeros([1,5])  #tem que somar o bias, por isso 4
rng.seed(22) #O valor da seed é modificado a cada teste, começando em 10
for i in range(5):
    w[0][i] = rng.uniform(-1,1)
    print("Peso " + str(i) + ": " + str(w[0][i]))



completedTrain = False
epocas = 0

while completedTrain == False:
    eaqm = SquareMeanError(x_train, w, y_train)    
    for i in range(trainQuantity):
        xb = np.hstack((bias, x_train.iloc[i].values))
        u = np.dot(w[0], xb)

        error = y_train.iloc[i] - u
        w[0] = w[0] + learnRate * error * xb
    epocas = epocas + 1
    eqmAtual = SquareMeanError(x_train, w, y_train)
    if np.absolute(eqmAtual - eaqm) <= precision:
        completedTrain = True

print("Terminou em " + str(epocas) + " epocas")
print("Pesos: " + str(w))
print ("Iniciando teste\n\n")

print("x0\tx1\tx2\tx3\ty\tlabel\t")
for i in range(testQuantity):
    label = y_test.iloc[i]
    xb = np.hstack((bias, x_test.iloc[i].values))
    v = np.dot(w[0], xb)
    y = activation(v)

    print(str(x_test.iloc[i][0]) + "\t" + str(x_test.iloc[i][1]) + "\t" + str(x_test.iloc[i][2])+ "\t" + str(x_test.iloc[i][3]) + "\t" + GetClass(y) + "\t" + GetClass(label))


print ("\nTestes dados extra:")
extra_data = [[0.9694, 0.6909, 0.4334, 3.4965], [0.5427, 1.3832,0.6390, 4.0352], [0.6081,-0.9196,0.5925, 0.1016],[-0.1618,0.4694,0.2030, 3.0117],[0.1870,-0.2578,0.6124, 1.7749], [0.4891,-0.5276,0.4378, 0.6439],[0.3777,2.0149,0.7423, 3.3932], [1.1498,-0.4067,0.2469,1.5866], [0.9325,1.0950,1.0359, 3.3591], [0.5060,1.3317,0.9222,3.7174], [0.0497,-2.0656,0.6124,-0.6585], [0.4004,3.5369,0.9766,5.3532], [-0.1874,1.3343,0.5374,3.2189], [0.5060,1.3317,0.9222,3.7174], [1.6375,-0.7911,0.7537,0.5515]]
for i in range(len(extra_data)):
    xb = np.hstack((bias, extra_data[i]))
    v = np.dot(w[0], xb)
    y = activation(v)

    print(str(extra_data[i][0]) + "\t" + str(extra_data[i][1]) + "\t" + str(extra_data[i][2])+ "\t" + str(extra_data[i][3]) + "\t" + GetClass(y))

f.close()