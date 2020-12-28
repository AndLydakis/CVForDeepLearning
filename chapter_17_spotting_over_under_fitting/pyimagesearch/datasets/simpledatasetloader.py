import numpy as np
import cv2
import os
from tqdm import tqdm


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths):
        # initialize list of features and labels
        data = []
        labels = []

        # loop over input images
        for i, imagePath in enumerate(tqdm(imagePaths)):
            # load the image and extract the class label assuming the format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]
            # preprocess image
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # save image
            data.append(image)
            labels.append(label)
        return np.array(data), np.array(labels)