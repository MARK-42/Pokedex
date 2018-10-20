from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

%matplotlib inline

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time

import keras
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers import Merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.advanced_activations import ELU, LeakyReLU, ThresholdedReLU

from keras.callbacks import ProgbarLogger, ModelCheckpoint

from PIL import Image

BATCH_SIZE = 32
EPOCHS = 8

target_size = (256,256)
grayscale = True

# Relative path for the train, test, and submission file
train_path = '/home/shashwat/input/train.csv'
test_path = '/home/shashwat/input/test.csv'

def load_image(id):
    img_path = '/home/shashwat/images/%d.jpg' % (id, )
    img = image.load_img(img_path,
                         grayscale=grayscale)
    img.thumbnail(target_size)
    bg = Image.new('L', target_size, (0,))
    bg.paste(
        img, (int((target_size[0] - img.size[0]) / 2), int((target_size[1] - img.size[1]) / 2))
    )
    img_arr = image.img_to_array(bg)
    
    return img_arr

# Load training data
train_data = pd.read_csv(train_path)
# load the ids in the training data set
x_ids = train_data.iloc[:, 0]
x_images = list()
for i in x_ids:
    x_images.append(load_image(i))
x_images = np.array(x_images)
x_features = train_data.iloc[:, 2:].values
# Convert the species to category type
y = train_data['species']
yy=y
# Get the corresponding categories list for species
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
#dictionary
dict={}
for i in range(990):
    index=y[i]
    name=yy[i]
    dict.update({index:name})

nb_classes = len(le.classes_)
# convert a class vectors (integers) to a binary class matrix
y= np_utils.to_categorical(y)
# Load testing data
test_data = pd.read_csv(test_path)
test_ids = test_data.iloc[:, 0]
test_images = list()
for i in test_ids:
    test_images.append(load_image(i))
test_images = np.array(test_images)

sss = StratifiedShuffleSplit(10, 0.2, random_state=15)
for train_index, test_index in sss.split(x_images, y):
	x_train_images, x_test_images, x_train_features, x_test_features = x_images[train_index], x_images[test_index], x_features[train_index], x_features[test_index]
	y_train, y_test = y[train_index], y[test_index]  

def construct_feature_model():
    print('Contructing the model')
    model=Sequential()
    model1 = Sequential([
       Dense(nb_classes * 2, input_shape=x_train_features.shape[1:]),
        BatchNormalization(),
        Activation('tanh'),
       Dropout(0.25)
    ])    
    model2 = Sequential([
        Convolution2D(32, 3, 3, border_mode='same',
                            input_shape=x_images.shape[1:]),
        Activation('tanh'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(nb_classes),
        Activation('tanh')
    ])    
    model = Sequential([
        Merge([model1, model2], mode='concat', concat_axis=1),
        Dense(nb_classes * 2),
        Activation('tanh'),
        Dropout(0.5),
        Dense(nb_classes),
        Activation('softmax')
    ])    
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)    
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])    
    return model

def train(model, x_images,x_features, x_test_images,x_test_features, y_test,y):
    
    model.fit([x_features, x_images], y, \
              batch_size=BATCH_SIZE, \
              epochs=EPOCHS, \
              verbose=1, \
              validation_data=([x_test_features, x_test_images], y_test))


model = construct_feature_model()

train(model, x_images,x_features, x_test_images,x_test_features, y_test,y)

y_prob = model.predict([test_data.iloc[:, 1:].values, test_images])

ans=['']*len(y_prob)
for i in range(len(y_prob)):
    index=0
    for j in range(99):
        if(y_prob[i][j]<y_prob[i][index]):
            index=j
    ans[i]=dict.get(index)        