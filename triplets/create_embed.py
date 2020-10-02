from os.path import join, exists
from os import listdir, makedirs
import sys
from matplotlib import pyplot as plt
import numpy as np
import os
import h5py
import cv2
from keras_facenet import FaceNet


# shuffling is required everytime you use these embeddings, since h5 files are created sequentially




trainX , testX , valX = [] , [] , []
trainY , testY , valY = [] , [] , []
counter = 0
facenet = FaceNet()

for dataset in ["train","test","val"]:
    for ind,label in enumerate(["manipulated_sequences","original_sequences"]):
        data_path = os.path.join("data",dataset,label)
        for img_name in os.listdir(data_path):
            img_path = os.path.join(data_path, img_name)

            img = cv2.imread(img_path)
            # print(img.shape)

            img = np.expand_dims(img, axis=0)
            # print(img.shape)

            embeddings = facenet.embeddings(img)
            # print(embeddings.shape)

            if dataset == "train":
                trainX.append(np.transpose(embeddings).flatten())
                trainY.append(ind)
            elif dataset == "test":
                testX.append(np.transpose(embeddings).flatten())
                testY.append(ind)
            else:
                valX.append(np.transpose(embeddings).flatten())
                valY.append(ind)
            
            counter += 1 

            sys.stdout.write('\r')
            sys.stdout.write(str(counter))
            sys.stdout.flush()



print()
trainX = np.array(trainX)
trainY = np.array(trainY)
trainY = trainY.reshape((len(trainY),1))

testX = np.array(testX)
testY = np.array(testY)
testY = testY.reshape((len(testY),1))

valX = np.array(valX)
valY = np.array(valY)
valY = valY.reshape((len(valY),1))


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
print(valX.shape)
print(valY.shape)





archive = h5py.File('train160.h5', 'w')
archive.create_dataset('/X', data = trainX)
archive.create_dataset('/Y',data = trainY)
archive.close()

archive = h5py.File('test160.h5', 'w')
archive.create_dataset('/X', data = testX)
archive.create_dataset('/Y',data = testY)
archive.close()

archive = h5py.File('val160.h5', 'w')
archive.create_dataset('/X', data = valX)
archive.create_dataset('/Y',data = valY)
archive.close()