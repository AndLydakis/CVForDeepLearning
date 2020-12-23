from pyimagesearch.nn.perceptron import Perceptron
import numpy as np
import argparse

# construct OR, AND, XOR datasets
X =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
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
    p = Perceptron(X.shape[1], alpha=0.1)
    p.fit(X, y, epochs=20)

    for (x, target) in zip(X, y):
        # make prediction
        pred = p.predict(x)
        print('[INFO] data={}, ground truth={}, pred={}'.format(x, target[0], pred))
