import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier


model = ResNet50(weights='imagenet')


# Read Data and split to half and extract features
def preprocess_data(Dataset_imgpath):
    X_train = []
    Y_train = []
    X_validation = []
    Y_validation = []
    img_val = []
    img_train = []
    i=1
    p=0
    k=0
    for path, _, files in os.walk(Dataset_imgpath):
        for file in files:
            l = len(files)
            if i <= l/2:
                    k=0
                    image = cv.imread(path + '\\' + file)
                    image = cv.resize(image, (224,224))
                    img_train.append(image)

                    x = np.array(image)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)

                    features = model.predict(x)

                    X_train.append(features)
                    label = path.split(os.path.sep)[-1]
                    Y_train.append(label)
                    p = p+1
                    i=i+1
            elif l/2 < i <= l:
                    p=0
                    image = cv.imread(path + '\\' + file)
                    image = cv.resize(image, (224,224))
                    img_val.append(image)
                    x = np.array(image)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)

                    features = model.predict(x)

                    X_validation.append(features)

                    label = path.split(os.path.sep)[-1]
                    Y_validation.append(label)
                    i=i+1
                    k = k+1
                    if i > l:
                        i=1
    Y_train = np.array(Y_train)
    X_train = np.array(X_train)

    Y_validation = np.array(Y_validation)
    X_validation = np.array(X_validation)
    return X_train,Y_train,X_validation,Y_validation



def apply_KNN(n_n,X_train,Y_train):
    Classifier = KNeighborsClassifier(n_neighbors = n_n,metric = 'minkowski', p=2)
    nsamples, nx, ny = X_train.shape
    d2_train_dataset = X_train.reshape((nsamples,nx*ny))
    Classifier.fit(d2_train_dataset,Y_train)


