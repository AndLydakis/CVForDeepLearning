from pyimagesearch.nn.neuralnetwork import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
data = digits.data.astype('float')
data = (data - data.min()) / (data.max() - data.min())
(trainX, testX, trainY, testY) = train_test_split(data, digits.target, test_size=0.25)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
training_losses = nn.fit(trainX, trainY, epochs=1000)

predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)

print(classification_report(testY.argmax(axis=1), predictions))

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 1000), training_losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()
