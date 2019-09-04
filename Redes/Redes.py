import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

f = open('out.txt', 'w')
sys.stdout = f

def activation(value):
    if value < 0.0:
        return -1
    else:
        return 1
def GetClass(value):
    if value == - 1:
        return "A"
    else:
        return "B"

data = pd.read_csv('dados.csv', sep=';')
maxEpocas = 1000

values = data[data.columns[0:3]]
label = data[data.columns[3]]

x_train, x_test, y_train, y_test = train_test_split(values, label, test_size = .3, random_state = 20)

trainQuantity = len(x_train)
testQuantity = len(x_test)

print("Iniciando treinamento com " + str(trainQuantity) + " amostras \n")
print("Entradas de treino \n" + x_train.to_string())
print("Labels de teste \n" + y_train.to_string())
print("Entradas de teste \n" + x_test.to_string())
print("Labels de treino \n" + y_test.to_string())

bias = 1
learnRate = 0.01
w = np.zeros([1,4])  #tem que somar o bias, por isso 4
errors = np.zeros(trainQuantity)



completedTrain = False
for j in range(maxEpocas):
    finished = True
    for i in range(trainQuantity):
        xb = np.hstack((bias, x_train.iloc[i].values))
        
        v = np.dot(w, xb)
        yr = activation(v)
        error = y_train.iloc[i] - yr
        errors[i] = error
        if error != 0:
            finished = False
        w = w + learnRate * errors[i] * xb
    if finished:
        print ("Esta sem erros na epoca: " + str(j))
        completedTrain = True
        break
print("Terminou por " + "não haver mais erros" if completedTrain else " número maximo de épocas")
print("Erros: " + str(errors))
print("Pesos: " + str(w))
print ("Iniciando teste\n\n")

print("x0\tx1\tx2\ty\tlabel")
for i in range(testQuantity):
    row = x_test.iloc[i]
    label = y_test.iloc[i]
    xb = np.hstack((bias, row.values))
    v = np.dot(w, xb)
    y = activation(v)

    print(str(row[0]) + "\t" + str(row[1]) + "\t" + str(row[2]) + "\t" + GetClass(y) + "\t" + GetClass(y))

f.close()