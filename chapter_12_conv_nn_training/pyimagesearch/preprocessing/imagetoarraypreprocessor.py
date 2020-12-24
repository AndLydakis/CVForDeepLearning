from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        # store the image data format
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply the Keras utility func to rearrange the dimensions of the image
        return img_to_array(image, data_format=self.dataFormat)
