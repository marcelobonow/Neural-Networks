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

data = pd.read_csv('dados.csv', sep=';')
maxEpocas = 1000

values = data[data.columns[0:3]]
label = data[data.columns[3]]

x_train, x_test, y_train, y_test = train_test_split(values, label, test_size = .3, random_state = 20)
trainQuantity = len(x_train)
testQuantity = len(x_test)

print("Iniciando treinamento com " + str(trainQuantity) + " amostras \n")
print("Entradas de treino \n" + x_train.to_string())
print("Labels de treino \n" + y_train.to_string())
print("Entradas de teste \n" + x_test.to_string())
print("Labels de teste \n" + y_test.to_string())

bias = -1
learnRate = 0.01
w = np.zeros([1,4])  #tem que somar o bias, por isso 4
rng.seed(10) #O valor da seed é modificado a cada teste, começando em 10
for i in range(4):
    w[0][i] = rng.uniform(-1,1)
    print("Peso " + str(i) + ": " + str(w[0][i]))
errors = np.zeros(trainQuantity)



completedTrain = False
for j in range(maxEpocas):
    finished = True
    for i in range(trainQuantity):
        xb = np.hstack((bias, x_train.iloc[i].values))
        
        v = np.dot(w[0], xb)
        yr = activation(v)
        error = y_train.iloc[i] - yr
        errors[i] = error
        if error != 0:
            finished = False
        w[0] = w[0] + learnRate * error * xb
    if finished:
        print ("Esta sem erros na epoca: " + str(j))
        completedTrain = True
        break
print("Terminou por " + "não haver mais erros" if completedTrain else " número maximo de épocas")
print("Erros: " + str(errors))
print("Pesos: " + str(w))
print ("Iniciando teste\n\n")

print("x0\tx1\tx2\ty\tlabel\t")
for i in range(testQuantity):
    label = y_test.iloc[i]
    xb = np.hstack((bias, x_test.iloc[i].values))
    v = np.dot(w[0], xb)
    y = activation(v)

    print(str(x_test.iloc[i][0]) + "\t" + str(x_test.iloc[i][1]) + "\t" + str(x_test.iloc[i][2]) + "\t" + GetClass(y) + "\t" + GetClass(label))

print ("\nTestes:")
extra_data = [[-0.3665, 0.0620, 5.9891], [-0.7842, 1.1267,5.5912], [0.3012,0.5611,5.8234],[0.7757,1.0648,8.0677],[0.1570,0.8028,6.3040], [-0.7014,1.0316,3.6005],[0.3748,0.1536,6.1537], [-0.6820,0.9404,4.4058], [-1.3970,0.7141,4.9263], [-1.8842,-0.2805,1.2548]]
for i in range(len(extra_data)):
    xb = np.hstack((bias, extra_data[i]))
    v = np.dot(w[0], xb)
    y = activation(v)

    print(str(extra_data[i][0]) + "\t" + str(extra_data[i][1]) + "\t" + str(extra_data[i][2]) + "\t" + GetClass(y))

f.close()