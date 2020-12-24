import numpy as np
from tqdm import tqdm


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # initialize the lsit of weight matrices
        self.W = []
        self.layers = layers
        self.alpha = alpha
        for i in np.arange(0, len(layers) - 2):
            # randomly initialize a weight matrix connecting the number of nodes
            # in each respective layer together, adding an extra node for the bias
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        # last two layers are a special case where the input connections need
        # a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # construct and return a string that represents the network architecture
        return 'Neural Network: {}'.format('-'.joint(str(l) for l in self.layers))

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # derivative of sigmoid function assuming x has passed through the sigmoid function
        return x * (1 - x)

    def fit_partial(self, x, y):
        # construct list of output activations for each layer as our data point flows through
        # the network: the first is a special case, it's just the input vector
        A = [np.atleast_2d(x)]
        # FEEDFORWARD
        for layer in np.arange(0, len(self.W)):
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            A.append(out)
        # BACKPROPAGATION
        # compute difference between prediction and ground truth
        error = A[-1] - y
        # apply chain rule and build list of deltas 'D'. first entry is just th error of the output layer
        # times the derivative of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is the delta of the previous layer
            # dotted with the Weight matrix of the current layer followed by
            # multiplying the delta by the derivative of the nonlinear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T) * self.sigmoid_deriv(A[layer])
            D.append(delta)
        # Reverse the deltas
        D = D[::-1]

        # Weight update phase
        for layer in np.arange(0, len(self.W)):
            # update weights by taking the dot product of the layer activations with their respective deltas
            # the multiplying this value by some small learning rate and adding to
            # our weight matrix -- this is where the actual learning happens
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # initialize the output prediction as the input features
        # this value will be propagated through the network to obtain the final prediction
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        # loop over layers
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    def calculate_loss(self, X, targets):
        # make prediction
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # insert the bias column
        X = np.c_[(X, np.ones((X.shape[0])))]

        losses = []
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            # check if we display
            loss = self.calculate_loss(X, y)
            losses.append(loss)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, loss))
        return losses
