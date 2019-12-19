from sklearn.neural_network import MLPClassifier



def RecognizeDigit(digit):
    mlp = MLPClassifier(hidden_layer_sizes=(1500, 1200), max_iter=10000,solver='sgd', verbose=True, tol=1e-4, learning_rate_init=0.0001)
    peso0 = np.array(pd.read_csv("peso0.csv", sep=','))
    peso1 = np.array(pd.read_csv("peso1.csv", sep=','))
    peso2 = np.array(pd.read_csv("peso2.csv", sep=','))
    return -1