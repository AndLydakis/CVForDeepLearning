from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
    ap.add_argument('-k', '--neighbors', type=int, default=1, help='# of nearest neighbors')
    ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of cores to use')
    args = vars(ap.parse_args())

    print('[INFO] KNN loading images ...')
    imagePaths = list(paths.list_images(args['dataset']))

    # initialize preprocessor
    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    data, labels = sdl.load(imagePaths)
    data = data.reshape((data.shape[0], 3072))

    print('[INFO] features matrix: {:.1f}MB'.format(data.nbytes / (1024 * 1000.0)))

    # encode labels as ints
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # partition dataset
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    # train and evaluate
    print('[INFO] evaluating k-NN classifier ...')
    model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
    model.fit(trainX, trainY)
    print(classification_report(testY, model.predict(testX), target_names=le.classes_))
