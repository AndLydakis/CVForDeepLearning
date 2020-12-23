import numpy as np
import cv2

if __name__ == '__main__':
    # initialize the class labels and set random seed
    labels = ['dog', 'cat', 'panda']
    np.random.seed(1)

    # randomly initialize weights and bias
    W = np.random.randn(3, 3072)  # 3 classes, 32x32x3 images
    b = np.random.randn(3)

    # load images
    orig = cv2.imread('beagle.jpg')
    image = cv2.resize(orig, (32, 32)).flatten()
    # compute output scores
    scores = W.dot(image) + b

    for (label, score) in zip(labels, scores):
        print('[INFO] {}: {:.2f}'.format(label, score))

    # draw label with highest score
    cv2.putText(orig,
                'Label: {}'.format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Labelled', orig)
    cv2.waitKey(0)
