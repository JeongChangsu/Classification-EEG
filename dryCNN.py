# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
# C:/Users/wjdck/PycharmProjects/pythonProject/CURTpy/dry_day1_featuremap.csv
binary_dataset = pd.read_csv("C:/Users/wjdck/PycharmProjects/pythonProject/CURTpy/dry_day1_featuremap.csv", header = None)
#binary_dataset = pd.read_csv("E:/Project/2019/WISET/Deeplearning/earECG.csv", header = None)
z= 0
A = np.zeros([14,5])
Recall = np.zeros([14,1])
# Specificity = np.zeros([14,1])
# adjust_accuracy = np.zeros([14,1])
# info = np.array([14,])
trial = [29,30,26,29,30,29,30,24,27,30,30,27,24,30]
print()

X = binary_dataset.values
Y = np.zeros([len(X),1])
a = len(X)

def build_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, strides=2, activation='relu', input_shape=(n_timestep, n_signal)))
    # model.add(Conv1D(filters= 2, kernel_size = 3, strides = 1, activation = 'relu', input_shape=(n_timestep, n_signal)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(units=20, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

n = 0
m = 0
for sess in  range(0,14):

    Y = np.zeros([len(X),1])
    temp = trial[sess]
    for sub in range(0,temp+1):
        Y[sub,:] = 1

    from random import shuffle
    ind_list = [i for i in range(len(X))]
    shuffle(ind_list)

    X = X[ind_list, :]
    Y = Y[ind_list, ]

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state =42)


    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Dropout, BatchNormalization
    from keras.layers.convolutional import Conv1D, MaxPooling1D

    X_tr = np.expand_dims(X_train, axis = 2)
    #x_tr = np.stack([X_train, X_train, X_train, X_train, X_train, X_train],axis = 1)

    n_timestep, n_signal = X_tr.shape[1], X_tr.shape[2]
    #n_timestep, n_signal = x_tr.shape[1], x_tr.shape[2]

    # CV
    import matplotlib.pyplot as plt

    # def plot_lost(history, fold):
    #     plt.figure(figsize=(6, 4))
    #     plt.plot(history.history['acc'], 'r', label = 'Accuracy of Training Data')
    #     plt.plot(history.history['val_acc'], 'b', label = 'Accuracy of Validation Data')
    #     plt.plot(history.history['loss'], 'r--', label = 'Loss of Training Data')
    #     plt.plot(history.history['val_loss'], 'b--', label = 'Loss of Validation Data')
    #     plt.title('Model Accuracy and Loss of fold')
    #     plt.ylabel('Accuracy and Loss')
    #     plt.xlabel('Training Epoch')
    #     plt.ylim(0)
    #     plt.legend()
    #     plt.show()

    # 5-fold cross-validation example
    kf = KFold(n_splits=5, shuffle = True)
    accuracy_list = []
    loss_list = []
    val_accuracy_list = []
    val_loss_list = []
    fold = 0

    for train_index, test_index in kf.split(X_tr):
        print("====================== Fold =======================")
        fold += 1
        print(fold)
        print("===================================================")

        model = build_model()
        history = model.fit(X_tr, Y_train, batch_size = 30, validation_split = 0.2, epochs = 100,  verbose = 1)

        accuracy = history.history['accuracy']
        accuracy_list.append('accuracy')

        loss = history.history['loss']
        loss_list.append(loss)

        val_accuracy = history.history['val_accuracy']
        val_accuracy_list.append(val_accuracy)

        val_loss = history.history['val_loss']
        val_loss_list.append(val_loss)

    average_accuracy_list = [np.mean([x[i] for x in accuracy_list]) for i in range(100)]
    average_val_accuracy_list = [np.mean([x[i] for x in val_accuracy_list]) for i in range(100)]
    average_loss_list = [np.mean([x[i] for x in val_accuracy_list]) for i in range(100)]
    average_val_loss_list = [np.mean([x[i] for x in val_loss_list]) for i in range(100)]

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(range(1, len(average_val_loss_list) + 1), average_val_loss_list, 'y', label = 'val loss')
    loss_ax.plot(range(1, len(average_loss_list) + 1), average_loss_list, 'r', label='loss')
    acc_ax.plot(range(1, len(average_val_accuracy_list) + 1), average_val_accuracy_list, 'b', label = 'val accuracy')
    acc_ax.plot(range(1, len(average_accuracy_list) + 1), average_accuracy_list, 'g', label='val accuracy')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuracy')

    X_te = np.expand_dims(X_test, axis = 2)

    test_loss, test_accuracy = model.evaluate(X_te, Y_test, batch_size = 25)

    Y_pred = model.predict(X_te)

    from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

    globals()['confusion_matrix_{}'.format(sess + 1)] = confusion_matrix(Y_test, Y_pred.round())

    # Recall[sess, 0] = recall_score(Y_test, Y_pred.round())
    # Specificity[sess, 1] = globals()['convusion_matrix_{}'.format(sess + 1)](1, 1) / (globals()['convusion_matrix_{}'.format(sess + 1)](1, 1) + globals()['convusion_matrix_{}'.format(sess + 1)](1, 2))
    #
    # adjust_accuracy[sess, 1] = (Recall[sess, 1] + Specificity[sess, 1]) / 2


    del model
    import os

    # os.system("pause")


    # n = n+(len(X)//14)
