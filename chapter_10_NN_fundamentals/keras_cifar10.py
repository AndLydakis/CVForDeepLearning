from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', required=True, help='path to outputplots', default='./cifar10')

    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX.astype('float') / 255.0
    testX = testX.astype('float') / 255.0
    trainX = trainX.reshape((trainX.shape[0], 3072))
    testX = testX.reshape((testX.shape[0], 3072))

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    labelNames = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)
    predictions = model.predict(testX, batch_size=128)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(x) for x in lb.classes_]))

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 100), H.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, 100), H.history['val_accuracy'], label='val_acc')
    plt.title('Training loss and accuracy')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
    plt.savefig(args['output'])
