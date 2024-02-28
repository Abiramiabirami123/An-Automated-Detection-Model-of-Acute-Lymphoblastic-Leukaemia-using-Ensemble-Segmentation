import numpy as np
import cv2 as cv
from keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from Evaluation import evaluation


def Model_RESNET(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [5, 5, 3]
    IMG_SIZE = [224, 224, 3]
    Activation = ['relu', 'sigmoid', 'linear', 'tanh']
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    base_model.add(int[sol[3]](units=train_target.shape[1], activation='sigmoid'))
    base_model.compile(loss='binary_crossentropy', metrics=['acc'], block_layers =int(sol[5]))
    try:
        base_model.fit(train_data, train_target, steps_per_epoch=10, epochs=int(sol[4]))
        weight = base_model.get_weights()[-1]  # features of the fully connected layer
        if sol:
            base_model.weights()[-1] = weight * sol  # Modified Transfer Learning
        pred = np.round(base_model.predict(test_data)).astype('int')
    except:
        pred = np.round(base_model.predict(test_data)).astype('int')
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        Eval = evaluation(pred, test_target)
        return Eval, pred
