
import numpy
import pandas
import sklearn
import sklearn.svm
import sklearn.neural_network


df = pandas.read_csv('SpiralWithCluster.csv')

total = 0
cluster1 = 0
for c in df['SpectralCluster']:
    if c == 1 or c == '1':
        cluster1 += 1
    total += 1

threshold = cluster1 / total
print('Opservations with SpectralCluster=1 : %s%%' % (100 * threshold)) # 50%



'''
b. (20 points) You will search for the neural network that yields the lowest loss value and the lowest misclassification rate.
    You will use your answer in (a) as the threshold for classifying an observation into SpectralCluster = 1.
    Your search will be done over a grid that is formed by cross-combining the following attributes:
        (1) activation function: identity, logistic, relu, and tanh;
        (2) number of hidden layers: 1, 2, 3, 4, and 5; and
        (3) number of neurons: 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10.
    List your optimal neural network for each activation function in a table.
    Your table will have four rows, one for each activation function.
    Your table will have six columns:
        (1) activation function,
        (2) number of layers,
        (3) number of neurons per layer,
        (4) number of iterations performed,
        (5) the loss value, and
        (6) the misclassification rate.
'''

#nn = sklearn.neural_network.MLPClassifier(learning_rate_init = 0.1, solver = 'lbfgs', random_state = 20200408, max_iter = 10000)

X = df[['x', 'y']]
Y = df['SpectralCluster']

def Build_NN_Class (actFunc, nLayer, nHiddenNeuron):

    # Build Neural Network
    nn = sklearn.neural_network.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                        activation = actFunc, verbose = False,
                        solver = 'lbfgs', learning_rate_init = 0.1,
                        max_iter = 10000, random_state = 20200408)
    fit = nn.fit(X, Y)

    # Test model
    pred = nn.predict_proba(X)

    #
    bad = 0
    for i in range(len(pred)):
        if pred[i] > threshold and Y[i] == 1:
            bad += 1
    return bad / len(pred)

    # Calculate Root Average Squared Error
    #    y_residual = Y - y_predProb
    #    rase = numpy.sqrt(numpy.mean(y_residual ** 2))
    return numpy.mean(y_predProb)

for i in numpy.arange(1,11):
    for j in numpy.arange(5,25,5):
        for act in ('identity', 'logistic', 'relu', 'tanh'):
           RASE = Build_NN_Class (actFunc = act, nLayer = i, nHiddenNeuron = j)
        #    print(i, j, act, RASE)
