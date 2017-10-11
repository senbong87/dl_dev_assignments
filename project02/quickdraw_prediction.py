import os
import glob
from itertools import groupby

import numpy as np
import tensorflow as tf

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
import keras.backend as K


def get_data(num_class, file_pattern="data/*.npy", train_samples_per_class=2**10, test_samples_per_class=2**8):
    # read raw data
    data_map = {}
    for i,f in enumerate(glob.glob(file_pattern)):
        if i < num_class:
            data_map[os.path.basename(f)] = np.load(f)

    train_data_map, test_data_map = {}, {}
    for key in data_map:
        train_data_map[key] = data_map[key].reshape((-1, 28, 28))[:,:,:,np.newaxis][:train_samples_per_class]
        test_data_map[key] = data_map[key].reshape((-1, 28, 28))[:,:,:,np.newaxis][train_samples_per_class:train_samples_per_class+test_samples_per_class]
        train_data_map[key] = train_data_map[key].astype("float32") / 255
        test_data_map[key] = test_data_map[key].astype("float32") / 255

    # construct train dataset
    class_orders = data_map.keys()
    train_data = np.concatenate([train_data_map[key] for key in class_orders])
    test_data = np.concatenate([test_data_map[key] for key in class_orders])
    train_target = np.concatenate([train_data_map[k].shape[0]*[i]
                                   for i,k in enumerate(class_orders)])
    test_target = np.concatenate([test_data_map[k].shape[0]*[i]
                                  for i,k in enumerate(class_orders)])
    train_target = np_utils.to_categorical(train_target)
    test_target = np_utils.to_categorical(test_target)
    return train_data, train_target, test_data, test_target


def param_generator(num_class):
    params = {'num_class': num_class}
    convs = [(64, 64, 64, 64, 64, 64)]
    dropouts = [(0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.05, 0.05)]
    batch_norms = [(True, True, True, True, True, True, True)]
    denses = [512]
    learning_rates = [0.03]
    decays = [0.4]
    for learning_rate in learning_rates:
        params["learning_rate"] = learning_rate
        for decay in decays:
            params["decay"] = decay
            for conv in convs:
                params["conv"] = conv
                for dense in denses:
                    params["dense"] = dense
                    for dropout in dropouts:
                        params["dropout"] = dropout
                        for batch_norm in batch_norms:
                            params["batch_norm"] = batch_norm
                            yield params


def get_model(params):
    input_layer = Input(shape=(28, 28, 1), name='Input_layer')

    # convblock 01
    x = Conv2D(params["conv"][0], (3, 3), padding='same', activation='relu',
            input_shape=input_layer.shape, name='Conv01_layer')(input_layer)
    x = Dropout(params["dropout"][0], name='Dropout01')(x)
    if params["batch_norm"][0]:
        x = BatchNormalization(name='BatchNormalization01_layer')(x)
    x = Conv2D(params["conv"][1], (3, 3), activation='relu', name='Conv02_layer')(x)
    x = Dropout(params["dropout"][1], name='Dropout02')(x)
    if params["batch_norm"][1]:
        x = BatchNormalization(name='BatchNormalization02_layer')(x)
    x = Conv2D(params["conv"][2], (3, 3), activation='relu', name='Conv03_layer')(x)
    x = Dropout(params["dropout"][2], name='Dropout03')(x)
    if params["batch_norm"][2]:
        x = BatchNormalization(name='BatchNormalization03_layer')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MaxPool01_layer')(x)

    # convblock 02
    x = Conv2D(params["conv"][3], (3, 3), padding='same', activation='relu',
            name='Conv04_layer')(x)
    x = Dropout(params["dropout"][3], name='Dropout04')(x)
    if params["batch_norm"][3]:
        x = BatchNormalization(name='BatchNormalization04_layer')(x)
    x = Conv2D(params["conv"][4], (3, 3), activation='relu', name='Conv05_layer')(x)
    x = Dropout(params["dropout"][4], name='Dropout05')(x)
    if params["batch_norm"][4]:
        x = BatchNormalization(name='BatchNormalization05_layer')(x)
    x = Conv2D(params["conv"][2], (3, 3), activation='relu', name='Conv06_layer')(x)
    x = Dropout(params["dropout"][5], name='Dropout06')(x)
    if params["batch_norm"][5]:
        x = BatchNormalization(name='BatchNormalization06_layer')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='MaxPool02_layer')(x)

    # fully connected dense block
    x = Flatten(name='Flatten_layer')(x)
    x = Dense(params["dense"], activation='relu', name='Dense01_layer')(x)
    x = Dropout(params["dropout"][6], name='Dropout07')(x)
    if params["batch_norm"][6]:
        x = BatchNormalization(name='BatchNormalization07_layer')(x)
    x = Dense(params["num_class"], name='logits_layer')(x)
    x = Dropout(params["dropout"][7], name='Dropout08')(x)
    output = Activation('softmax', name='Softmax_layer')(x)

    # construct model
    model = Model(input_layer, output)
    return model


def _tuple2str(t):
    return "-".join("{0}p{1}".format(len(list(g)), i) for i,g in groupby(t))


def construct_key(params):
    key = "lr{0}-decay{1}-conv{2}-dense{3}-drop{4}-batch{5}".format(
            params["learning_rate"], params["decay"], _tuple2str(params["conv"]), params["dense"],
            _tuple2str(params["dropout"]), _tuple2str(params["batch_norm"]))
    return key


if __name__ == '__main__':
    
    # define parameters
    NUM_CLASS = 100
    BATCH_SIZE = 2**8
    NUM_TRAIN_SAMPLES_PER_CLASS = 2**10
    NUM_EPOCH = 200
    LOGDIR = "./logs"
    
    # construct train data generator
    train_data, train_target, test_data, test_target = get_data(NUM_CLASS, train_samples_per_class=NUM_TRAIN_SAMPLES_PER_CLASS)
    train_data_gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                        height_shift_range=0.1, horizontal_flip=True,
                                        vertical_flip=False)
    train_data_gen.fit(train_data)

    # iterates through different model & learning parameters
    for params in param_generator(NUM_CLASS):
        key = construct_key(params)
        model = get_model(params)
        callbacks = [EarlyStopping(monitor='loss', min_delta=0.001, patience=10),
                     LearningRateScheduler(lambda epoch: params["learning_rate"]/(1 + params["decay"] * epoch)),
                     ReduceLROnPlateau(monitor='loss'),
                     TensorBoard(log_dir=os.path.join(LOGDIR, key))]
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        K.set_value(model.optimizer.lr, params["learning_rate"])
        model.fit_generator(train_data_gen.flow(train_data, train_target, batch_size=BATCH_SIZE),
                            steps_per_epoch=train_data.shape[0] // BATCH_SIZE,
                            validation_data=(test_data, test_target),
                            epochs=NUM_EPOCH,
                            workers=8,
                            callbacks=callbacks)
