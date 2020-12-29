import argparse
import imutils
import cv2
import os

from imutils import paths
from tqdm import tqdm

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='path to download images')
    ap.add_argument('-a', '--annot', required=True, help='path to annotated images')
    args = vars(ap.parse_args())

    imagePaths = list(paths.list_images(args['input']))
    counts = {}

    for i, imagePath in enumerate(tqdm(imagePaths)):
        try:
            print(imagePath)
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

            thres = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # find contours, keep the largest 4
            cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[1] if imutils.is_cv2() else cnts[0]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

                cv2.imshow('ROI', imutils.resize(roi, width=28))
                key = cv2.waitKey(0)

                if key == ord('~'):
                    continue

                key = chr(key).upper()
                dirPath = os.path.sep.join([args['annot'], key])
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)

                count = counts.get(key, 1)
                p = os.path.sep.join([dirPath, '{}.png'.format(str(count).zfill(6))])
                cv2.imwrite(p, roi)
                counts[key] = count + 1
        except KeyboardInterrupt as e:
            print(e)
            break
        except KeyboardInterrupt as e:
            print('Skipping image')
