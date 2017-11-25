from __future__ import print_function

import os
import threading
import multiprocessing

import pandas as pd
from keras.models import Model
from keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras.applications.xception import Xception
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback, ModelCheckpoint
import keras.backend as K

from util.generator import BSONIterator


def get_model(num_class):
    # define model architecture
    base_model = Xception(weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool1")(x)
    x = Dense(1024, activation="relu", name="relu1")(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu", name="relu2")(x)
    x = Dropout(0.05, name="dropout1")(x)
    predictions = Dense(num_class, activation="softmax", name="output")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all convolutional InceptionV3 layer
    for i, layer in enumerate(base_model.layers):
        if i < 50:
            layer.trainable = False

    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':

    # define global variables
    LOCK = threading.Lock()
    DATA_DIR = "./data"
    MODEL_FILEPATH = "./.model_checkpts/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    TRAIN_BSON_FILE = open(os.path.join(DATA_DIR, "raw/train.bson"), "rb")
    CPU_COUNT = multiprocessing.cpu_count()

    # model parameters
    NUM_CLASS = 5270

    # define learning parameters
    BATCH_SIZE = 128
    LR = 0.001
    DC = 0.01
    SPLIT_FACTOR = 5
    EPOCH = 10

    # load dataframe
    train_offset_df = pd.read_csv(os.path.join(DATA_DIR, "processed/train_offset.csv"), index_col="product_id")
    train_image_df = pd.read_csv(os.path.join(DATA_DIR, "processed/train_img.csv"))
    valid_image_df = pd.read_csv(os.path.join(DATA_DIR, "processed/valid_img.csv"))

    # prepare training & validation generators
    train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                                    height_shift_range=0.2, zoom_range=0.2,
                                    horizontal_flip=True, fill_mode="nearest")
    train_generator = BSONIterator(TRAIN_BSON_FILE, train_image_df, train_offset_df,
                                NUM_CLASS, train_datagen, LOCK, batch_size=BATCH_SIZE,
                                shuffle=True)
    valid_datagen = ImageDataGenerator()
    valid_generator = BSONIterator(TRAIN_BSON_FILE, valid_image_df, train_offset_df,
                                NUM_CLASS, valid_datagen, LOCK, batch_size=valid_image_df.shape[0])
    X_valid, y_valid = next(valid_generator)

    # get the cnn model
    model = get_model(NUM_CLASS)
    model.summary()

    # prepare callback functions
    callbacks = [EarlyStopping(monitor="loss", min_delta=0.001, patience=3),
                 ReduceLROnPlateau(monitor="loss"),
                 LearningRateScheduler(lambda epoch: LR/(1 + DC*epoch)),
                 LambdaCallback(on_epoch_begin=lambda e,l: print("Learning rate:", K.get_value(model.optimizer.lr))),
                 ModelCheckpoint(filepath=MODEL_FILEPATH, verbose=0, period=1)]

    # train the model
    print("CPU count:", CPU_COUNT)
    K.set_value(model.optimizer.lr, LR)
    model.fit_generator(train_generator, steps_per_epoch=train_image_df.shape[0]//(SPLIT_FACTOR*BATCH_SIZE),
                        epochs=SPLIT_FACTOR*EPOCH,
                        validation_data=(X_valid, y_valid),
                        workers=CPU_COUNT,
                        callbacks=callbacks)
