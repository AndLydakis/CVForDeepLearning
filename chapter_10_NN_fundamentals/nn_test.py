from pyimagesearch.nn.neuralnetwork import NeuralNetwork
import numpy as np
import argparse

# construct OR, AND, XOR datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
datasets = {
    'or': {'y': np.array([[0], [1], [1], [1]])},
    'and': {'y': np.array([[0], [0], [0], [1]])},
    'xor': {'y': np.array([[0], [1], [1], [0]])}
}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', help='or, and, xor', required=True)
    args = vars(ap.parse_args())
    # define our perceptron and train it
    y = datasets[args['dataset']]['y']
    nn = NeuralNetwork([2, 2, 1], alpha=0.5)
    nn.fit(X, y, epochs=2000)

    for x, target in zip(X, y):
        pred = nn.predict(x)[0][0]
        step = 1 if pred > 0.5 else 0
        print('[INFO] data = {}, ground-truth={}, pred={:.4f}, step={}'.format(x, target[0], pred, step))
