from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import os
import numpy

MODEL_SAVE_FOLDER_PATH = './model/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)

    model_path = MODEL_SAVE_FOLDER_PATH + 'mnist-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

    cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                    verbose=1, save_best_only=True)

    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    (X_train, Y_train), (X_validation, Y_validation) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
    X_validation = X_validation.reshape(X_validation.shape[0], 28, 28, 1).astype('float32') / 255

    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_validation = np_utils.to_categorical(Y_validation, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                                metrics=['accuracy'])

    history = model.fit(X_train, Y_train,
                        validation_data=(X_validation, Y_validation),
                                            epochs=3, batch_size=200, verbose=0,
                                                                callbacks=[cb_checkpoint, cb_early_stopping])

    print('\nAccuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation)[1]))

    y_vloss = history.history['val_loss']
    y_loss = history.history['loss']

    x_len = numpy.arange(len(y_loss))
    plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
    plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")

    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
