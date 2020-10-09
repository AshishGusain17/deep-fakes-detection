from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K
from keras import applications, Sequential, optimizers
from keras.layers import Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import numpy as np
import time
from os.path import exists
from os import makedirs
import os






IMG_SIZE  = 160
base_model =   applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
print(base_model.input , base_model.output)

add_model = base_model.output
add_model = Flatten()(add_model)
add_model = Dropout(0.5)(add_model)
add_model = Dense(512, activation='relu')(add_model)
add_model = Dropout(0.4)(add_model)
add_model = Dense(256, activation='relu')(add_model)
add_model = Dropout(0.4)(add_model)
add_model = Dense(128, activation='relu')(add_model)
add_model = Dropout(0.4)(add_model)
add_model = Dense(2, activation='softmax')(add_model)

model = Model(inputs = base_model.input, outputs = add_model)
model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),metrics=['accuracy'])
model.summary()








train_folder = "data/train/"
test_folder = "data/test/"
val_folder = "data/val/"







datagenTrain = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    # rotation_range=30,
    # zoom_range=0.15,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.15,
    # fill_mode="nearest",
)
datagenTest = ImageDataGenerator(rescale=1.0 / 255.0)
datagenVal = ImageDataGenerator(rescale=1.0 / 255.0)

# prepare an iterators for each dataset
train_it = datagenTrain.flow_from_directory(train_folder,
                                            target_size=(160, 160), 
                                            batch_size = 64,
                                            class_mode='binary')
test_it = datagenTest.flow_from_directory(test_folder,
                                          target_size=(160, 160), 
                                          batch_size = 64,
                                          class_mode='binary')
val_it = datagenVal.flow_from_directory(val_folder,
                                          target_size=(160, 160), 
                                          batch_size = 64,
                                          class_mode='binary')

# confirm the iterator works
train_X, train_Y = train_it.next()
print('Batch shape=%s, %s, min=%.3f, max=%.3f' % (train_X.shape, train_Y.shape, train_X.min(), train_X.max()))

test_X, test_Y = test_it.next()
print('Batch shape=%s, %s, min=%.3f, max=%.3f' % (test_X.shape, test_Y.shape, test_X.min(), test_X.max()))

val_X, val_Y = val_it.next()
print('Batch shape=%s, %s, min=%.3f, max=%.3f' % (val_X.shape, val_Y.shape, val_X.min(), val_X.max()))














steps_per_epoch = train_it.n//train_it.batch_size
print(train_it.n, train_it.batch_size, steps_per_epoch)

validation_steps  = val_it.n//val_it.batch_size
print(val_it.n,   val_it.batch_size,   validation_steps)

history = model.fit_generator(generator = train_it,
                    steps_per_epoch = steps_per_epoch,
                    validation_data = val_it,
                    validation_steps = validation_steps,
                    epochs=100,
                    verbose = 1
)



