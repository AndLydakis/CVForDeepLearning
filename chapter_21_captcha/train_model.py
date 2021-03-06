import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
from pyimagesearch.nn.conv.lenet import LeNet
from pyimagesearch.nn.utils.captchahelper import preprocess
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from imutils import paths

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
    ap.add_argument('-m', '--model', required=True, help='path to output model')
    args = vars(ap.parse_args())

    data = []
    labels = []

    for imagePath in paths.list_images(args['dataset']):
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = preprocess(image, 28, 28)
        image = img_to_array(image)
        data.append(image)
        labels.append(imagePath.split(os.path.sep)[-2])

    data = np.array(data, dtype='float') / 255.0
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels,
                                                      test_size=0.25,
                                                      random_state=42)

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    opt = SGD(lr=0.01)
    num_classes = np.unique(labels).shape[0]
    model = LeNet.build(width=28, height=28, depth=1, classes=num_classes)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=20, verbose=1)

    predictions = model.predict(testX, batch_size=32)

    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=[str(x) for x in lb.classes_]))

    model.save(args['model'])

    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, 20), H.history['loss'], label='train_loss')
    plt.plot(np.arange(0, 20), H.history['val_loss'], label='val_loss')
    plt.plot(np.arange(0, 20), H.history['accuracy'], label='acc')
    plt.plot(np.arange(0, 20), H.history['val_accuracy'], label='val_acc')

    plt.title('Training Loss and Accuracy')
    plt.xlabel('Epoch #')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
