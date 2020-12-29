import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, N, alpha=0.1):
        # initializa the weight matrix and store the learning rate
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, X):
        # apply step function
        return 1 if X > 0 else 0

    def fit(self, X, y, epochs=10):
        # insert a colun of 1s for trainable bias
        X = np.c_[X, np.ones((X.shape[0]))]
        # loop over epochs
        for epoch in tqdm(range(epochs)):
            for (x, target) in zip(X, y):
                # take dot product between input and weight, the pass the value through the step function
                p = self.step(np.dot(x, self.W))

                # perform weight upgrade if our prediction does not match the target
                if p != target:
                    # determine error
                    error = p - target
                    # update weight matrix
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):
        # ensure input is matrix
        X = np.atleast_2d(X)
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.W))
