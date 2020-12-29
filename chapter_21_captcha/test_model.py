# import the necessary packages
import argparse
import imutils
import os
import cv2
import numpy as np

from imutils import paths
from imutils import contours
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from pyimagesearch.nn.utils.captchahelper import preprocess

if __name__ == '__main__':
    # Define input arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input directory of images")
    ap.add_argument("-m", "--model", required=True,
                    help="path to the model")
    args = vars(ap.parse_args())

    # Load the model and the label binarizer
    model = load_model(args["model"])

    # loop over image paths
    image_paths = list(paths.list_images(args['input']))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.copyMakeBorder(image, 25, 0, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20,
                                  cv2.BORDER_REPLICATE)

        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = cnts[0]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
        cnts = contours.sort_contours(cnts)[0]
        # initialize the output image as a "grayscale" image with 3
        # channels along with the output predictions
        output = cv2.merge([gray] * 3)
        predictions = []

        for c in cnts:
            # compute the bounding box for the contour then extract the
            # digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            # pre-process the ROI and classify it then classify it
            # pre-process the character and classify it
            roi = preprocess(roi, 28, 28)
            roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0

            pred = model.predict(roi).argmax(axis=1)[0] + 1
            predictions.append(str(pred))
            # draw the prediction on the output image
            cv2.rectangle(output, (x - 2, y - 2),
                          (x + w + 4, y + h + 4), (0, 255, 0), 1)
            cv2.putText(output, str(pred), (x - 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # draw the prediction on the output image
        print("[INFO] captcha: {}".format(",".join(predictions)))
        cv2.imshow("Output", output)
        cv2.waitKey()
