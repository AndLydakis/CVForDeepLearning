from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import argparse


def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    # take the dot product between our features and weight matrix
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds


def next_batch(X, y, batchSize):
    # loop over our dataset X in mini-batches
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i: i + batchSize], y[i: i + batchSize])


ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=int, default=100, help='# of epochs')
ap.add_argument('-a', '--alpha', type=int, default=100, help='learning rate')
ap.add_argument('-m', '--method', type=str, default='sgd', help='gradient descent variant')
ap.add_argument('-n', '--batch-size', type=int, default=32, help='batch size for SGD mini-batches')
args = vars(ap.parse_args())

# create a 2 class classification proble with 1000 data points, each with a 2d feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1s as the last entry in the feature matrix (treat bias vector as trainable param)
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of the data
# for training and for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print('[INFO] Training ...')

W = np.random.randn(X.shape[1], 1)
losses = []


print('Method: {}'.format(args['method']))

t = trange(args['epochs'], desc='[INFO] epoch={}, loss={:.7f}'.format(0, 0), leave=True)
for epoch in tqdm(np.arange(0, args['epochs'])):

    # Use mini batch stochastic gradient descent
    if args['method'] == 'sgd':
        epochLoss = []
        for (batchX, batchY) in next_batch(X, y, args['batch_size']):
            preds = sigmoid_activation(batchX.dot(W))
            error = preds - batchY
            epochLoss.append(np.sum(error ** 2))
            gradient = batchX.T.dot(error)

            W += -args['alpha'] * gradient

        loss = np.average(epochLoss)
        losses.append(loss)
    else:
        # take dot product between X and W, pass through sigmoid activation function
        preds = sigmoid_activation(trainX.dot(W))

        # now that we have our predictions we need to determine the error
        # which is the difference between our predictions and the true values
        error = preds - trainY
        loss = np.sum(error ** 2)
        losses.append(loss)

        # gradient descent update is the dot product between our features and the error of the prediction
        gradient = trainX.T.dot(error)

        # in the update stage, we push the weights in the negative direction of the gradient by taking
        # a small step towards a set of more optimal parameters
        W += -args['alpha'] * gradient

    # check to see if an update should be displayed
    t.set_description('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, losses[-1]))
    print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, losses[-1]))

print('[INFO] evaluating ...')
preds = predict(testX, W)

print(classification_report(testY, preds))

# plot the testing classification data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, s=30)

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
