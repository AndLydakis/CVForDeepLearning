import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn import datasets
from pyimagesearch.nn.conv.minivggnet import MiniVGGNet
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.datasets import cifar10


def step_decay(epoch, factor=0.25, dropEvery=5):
    initAlpha = 0.01

    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    return float(alpha)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--output', type=str, help='path to out plot', required=True)
    ap.add_argument('-n', '--norm', type=int, default=0, help='normalize or not (0/1)', required=True)
    ap.add_argument('-d', '--decay', type=int, default=-1, help='decay: -1 None, 0 step, 1 custom', required=True)
    ap.add_argument('-w', '--weights', type=str, default=-1, help='path to model checkpoint', required=True)
    args = vars(ap.parse_args())

    ((trainX, trainY), (testX, testY)) = cifar10.load_data()
    trainX = trainX.astype('float') / 255.0
    testX = testX.astype('float') / 255.0

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    labelNames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    callbacks = []
    if args['decay'] == 0:
        callbacks = [LearningRateScheduler(step_decay)]
    if args['decay'] == 1:
        curpath = os.path.abspath(os.curdir)
        figPath = curpath + os.path.sep.join([args['output'], str(os.getpid()), 'plot.png'])
        jsonPath = curpath + os.path.sep.join([args['output'], str(os.getpid()), 'history.json'.format(os.getpid())])
        if not os.path.exists(os.path.dirname(figPath)):
            try:
                os.makedirs(os.path.dirname(figPath))
            except Exception as exc:
                raise
        callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

    # Use templated string to save all models
    # cpPath = curpath + os.path.sep.join(
    #     [args['output'], args['weights'] + '/{}.hdf5'.format(os.getpid()), 'weights-{epoch:03d}-{val_loss:.4f}.hdf5'])
    # if not os.path.exists(os.path.dirname(cpPath)):
    #     try:
    #         os.makedirs(os.path.dirname(cpPath))
    #     except Exception as exc:
    #         raise

    cpPath = curpath + os.path.sep.join([args['output'], str(os.getpid()), args['weights'] + '.hdf5'])
    print('--------------------')
    print(figPath)
    print(jsonPath)
    print(cpPath)
    print('--------------------')
    checkpoint = ModelCheckpoint(cpPath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    callbacks.append(checkpoint)

    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10, normalize=args['norm'])
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1,
                  callbacks=callbacks)

    predictions = model.predict(testX, batch_size=64)

    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 40), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 40), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 40), H.history['accuracy'], label='train_acc')
    plt.plot(np.arange(0, 40), H.history['val_accuracy'], label='val_acc')

    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
