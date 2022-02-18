import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from read_data import MacOSFile


class KNNDigitClassifier:

    def __init__(self, path='plates_dataset/CNN letter Dataset', batch_size=(180, 120)):
        self.path = path
        self.batch_size = batch_size
        self.categories = [c for c in os.listdir(path) if
                           c not in ['.DS_Store', 'binarized']]
        self.model = KNeighborsClassifier(n_neighbors=len(self.categories), n_jobs=5)
        for c in self.categories:
            new_path = os.path.join(path, 'binarized', c)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
        # train_preprocess (run only for the first time on new data

    def train_preprocess(self):
        for c in self.categories:
            new_path = os.path.join(self.path, 'binarized', c)
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
        for i, l in enumerate(self.categories):
            images = os.listdir(os.path.join(self.path, l))
            for img in images:
                file_path = os.path.join(self.path, l, img)
                if os.path.isfile(file_path):
                    im = cv2.imread(file_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    th, im = cv2.threshold(im, 100, 255, cv2.THRESH_OTSU)
                    if not cv2.imwrite(os.path.join(self.path, 'binarized', l, img), im):
                        raise Exception("Could not write image")

    def train(self):
        data, labels = [], []
        binary_path = os.path.join(self.path, 'binarized')
        for i, l in enumerate(self.categories):
            images = os.listdir(os.path.join(binary_path, l))
            for img in images:
                # print(f'{l} {img}')
                file_path = os.path.join(binary_path, l, img)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                label = file_path.split(os.path.sep)[-2].split(".")[0]
                pixels = cv2.resize(image, self.batch_size).flatten()
                data.append(pixels)
                labels.append(label)
        data = np.array(data)
        labels = np.array(labels)
        (trainI, testI, trainL, testL) = train_test_split(
            data, labels, test_size=0.9, random_state=42)
        self.model.fit(trainI, trainL)
        #self.model.score(testI, testL)
        with open('my_dumped_classifier.pkl', 'wb') as fid:
            pickle.dump(self.model, MacOSFile(fid), protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self):
        with open('my_dumped_classifier.pkl', "rb") as f:
            self.model = pickle.load(MacOSFile(f))

    def preprocess_data(self, array):
        fig = plt.figure(figsize=(14, 4))
        grid = gridspec.GridSpec(ncols=len(array),
                                 nrows=1,
                                 figure=fig)
        for i, l in enumerate(array):
            fig.add_subplot(grid[i])
            plt.axis(False)
            m = np.invert(l)
            if m.any():
                kernel = np.ones((5, 5), np.uint8)
                n = cv2.dilate(m, kernel)
                n = cv2.resize(n, self.batch_size)
                _, n = cv2.threshold(n, 127, 255, cv2.THRESH_OTSU)
                array[i] = n.flatten()
                plt.imshow(n, cmap='gray')
        plt.show()
        return np.array(array)

    def predict(self, array):
        array = self.preprocess_data(array)
        predictions = []
        for a in array:
            predictions.append(self.model.predict(a.reshape(1, -1)))
        return predictions
