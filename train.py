import matplotlib
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import argparse
import tensorflow as tf
from keras import backend as K
from imutils import paths
from os.path import join
import random
import cv2
import os
import sys
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from model import SRCNN
from test import psnr1

# num = 21884
EPOCHS = 6
INIT_LR = 1e-4
BS = 128
input_size = 33


def load_data(path):
    print("[INFO] loading images...")
    data = []
    labels= []
    imagePaths = list(paths.list_images(join(path,'train')))
    for imagePath in imagePaths:
        image = misc.imread(imagePath)[:,:,0,None]
        labelPath = join(path,'label',imagePath.split('/')[-1])
        label = misc.imread(labelPath)[:, :, 0, None]
        # print(imagePath,labelPath)
        labels.append(label)
        data.append(image)

    # data = np.array(data, dtype="float") / 255.0
    # labels = np.array(labels, dtype="float") / 255.0
    data = np.array(data, dtype="float")
    labels = np.array(labels, dtype="float")
    print(data.shape)
    return data, labels


def train_batch(aug, trainX, trainY,load_weight=True):
    print("[INFO] compiling model...")
    model = SRCNN()

    if load_weight and os.path.exists('my_model_weights.h5') :
        print("[INFO] import model weights...")
        model.load_weights('my_model_weights.h5')
    try:
        # opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        opt = Adam(lr=INIT_LR)
        model.compile(loss="mean_squared_error", optimizer=opt, metrics=[psnr1])
        current_time = datetime.now().strftime("%Y%m%d-%H:%M")
        filepath = 'train_model/model-ep{epoch:03d}-loss{loss:.3f}-psnr{psnr1:.3f}-'+ current_time + '.h5'

        checkpoint = ModelCheckpoint(filepath, monitor='psnr1', verbose=1, save_best_only=True,
                                     mode='max', period=1)
        print("[INFO] training network...")
        H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                                steps_per_epoch=len(trainX) // BS, epochs=EPOCHS,callbacks=[checkpoint])
        print("[INFO] training finish")

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupted...")

    except Exception as e:
        print(e)

    else:
        plt.figure()
        N = EPOCHS
        # plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        # plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["psnr1"], label="psnr")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper left")
        plt.savefig("plot.png")

    finally:
        print("[INFO] serializing network...")
        model.save("./model.h5")
        model.save_weights('my_model_weights.h5')


def train(trainX,trainY):
    m = SRCNN()
    m.compile(Adam(lr=0.0003), 'mse')
    count = 1
    while True:
        m.fit(trainX, trainY, batch_size=128, nb_epoch=5)
        print("Saving model " + str(count * 5))
        m.save(join('./model_' + str(count * 5) + '.h5'))
        count += 1



if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)
    train_file_path = "./train_data"
    trainX, trainY = load_data(train_file_path)
    aug = ImageDataGenerator()
    train_batch(aug, trainX, trainY,)
    # train(trainX,trainY)
