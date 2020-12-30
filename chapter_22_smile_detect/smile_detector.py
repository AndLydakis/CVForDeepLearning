from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.models import load_model
from pyimagesearch.nn.conv.lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
import os

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', default='./dataset/SMILEs', help='path to dataset')
    ap.add_argument('-m', '--model', default='./output/lenet.hdf5', help='path to output model')
    ap.add_argument('-t', '--test', type=int, default=0, help='set to 1 to also test model')
    ap.add_argument('-v', '--video', help='path to video file for testing')
    args = vars(ap.parse_args())

    if args['test'] == 1:
        detector = cv2.CascadeClassifier('cascade_class.xml')
        model = load_model(args['model'])
        if not args.get('video', False):
            print('[INFO] Using Camera Capture')
            camera = cv2.VideoCapture(0)
        else:
            camera = cv2.VideCapture(args['video'])

        while True:
            (grabbed, frame) = camera.read()

            if args.get('video') and not grabbed:
                print('[INFO] Could not get Video Frame')
                break

            frame = imutils.resize(frame, width=300)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameClone = frame.copy()

            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

            for (fX, fY, fW, fH) in rects:
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (28, 28))
                roi = roi.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                (notSmiling, smiling) = model.predict(roi)[0]
                label = 'Smiling' if smiling > notSmiling else 'Not Smiling'

                cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            cv2.imshow('Face', frameClone)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
    else:
        data = []
        labels = []
        for imagePath in sorted(list(paths.list_images(args['dataset']))):
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = imutils.resize(image, width=28)
            image = img_to_array(image)
            data.append(image)

            label = imagePath.split(os.path.sep)[-3]
            label = 'smiling' if label == 'positives' else 'not_smiling'
            labels.append(label)
        data = np.array(data, dtype='float') / 255.0
        labels = np.array(labels)

        le = LabelEncoder().fit(labels)
        labels = np_utils.to_categorical(le.transform(labels), 2)

        classTotals = labels.sum(axis=0)
        classWeight = classTotals.max() / classTotals
        print(classWeight)

        (trainX, testX, trainY, testY) = train_test_split(data,
                                                          labels,
                                                          test_size=0.25,
                                                          random_state=42)

        model = LeNet.build(width=28, height=28, depth=1, classes=2)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        H = model.fit(trainX, trainY, validation_data=(testX, testY),
                      class_weight={0: classWeight[0], 1: classWeight[1]}, batch_size=64, epochs=20, verbose=1)

        predictions = model.predict(testX, batch_size=64)

        print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                    target_names=[str(x) for x in le.classes_]))

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
