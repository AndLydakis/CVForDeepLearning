import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import argparse

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from sklearn import datasets
from keras.datasets import cifar10


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', help='path to out plot', required=True)
    ap.add_argument('-n', '--norm', type=int, default=0, help='normalize or not (0/1)', required=True)
    args = vars(ap.parse_args())

    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype('float') / 255.0
    testX = testX.astype('float') / 255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    labelNames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10, normalize=args['norm'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)

    predictions = model.predict(testX, batch_size=64)

    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 40), H.history['accuracy'], label='acc')
    plt.plot(np.arange(0, 40), H.history['val_accuracy'], label='val_acc')

    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(args['output'])
