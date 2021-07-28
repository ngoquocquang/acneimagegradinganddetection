import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
import os
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from resnet import Model_train
from stacking import load_all_models, define_stacked_model, fit_stacked_model, predict_stacked_model
import tensorflow as tf

epochs = 80
batch_size = 32
batch_size_val = 16
path_image = "dataset/JPEGImages"
path_train = "dataset/ImageSets/Main/NNEW_trainval_3.txt"
path_val = "dataset/ImageSets/Main/NNEW_test_3.txt"
load_model_path = "model/model_"
path_model="model/ensemble_model.h5"
checkpoint_filepath="model/checkpoint/ResNet_{epoch:02d}-{loss:.2f}.h5"
path = "model/model_2.h5"
n_members = 3
fp_train = open(path_train, 'r')
filenames_train = []
mild_train = []
moderate_train = []
severe_train = []
verysevere_train = []
lesions_train = []
for line in fp_train.readlines():
    filename_train, label_train, lesion_train = line.split()
    filenames_train.append(filename_train)
    if label_train == '0':
        mild_train.append(1)
        moderate_train.append(0)
        severe_train.append(0)
        verysevere_train.append(0)
    elif label_train == '1':
        mild_train.append(0)
        moderate_train.append(1)
        severe_train.append(0)
        verysevere_train.append(0)
    elif label_train == '2':
        mild_train.append(0)
        moderate_train.append(0)
        severe_train.append(1)
        verysevere_train.append(0)
    else:
        mild_train.append(0)
        moderate_train.append(0)
        severe_train.append(0)
        verysevere_train.append(1)
    lesions_train.append(lesion_train)
fp_train.close()

fp_val = open(path_val, 'r')
filenames_val = []
mild_val = []
moderate_val = []
severe_val = []
verysevere_val = []
lesions_val = []
for line in fp_val.readlines():
    filename_val, label_val, lesion_val = line.split()
    filenames_val.append(filename_val)
    if label_val == '0':
        mild_val.append(1)
        moderate_val.append(0)
        severe_val.append(0)
        verysevere_val.append(0)
    elif label_val == '1':
        mild_val.append(0)
        moderate_val.append(1)
        severe_val.append(0)
        verysevere_val.append(0)
    elif label_val == '2':
        mild_val.append(0)
        moderate_val.append(0)
        severe_val.append(1)
        verysevere_val.append(0)
    else:
        mild_val.append(0)
        moderate_val.append(0)
        severe_val.append(0)
        verysevere_val.append(1)
    lesions_val.append(lesion_val)
fp_val.close()

train_df = pd.DataFrame({
    'filename_train': filenames_train,
    'mild': mild_train,
    'moderate': moderate_train,
    'severe': severe_train,
    'verysevere': verysevere_train
})

val_df = pd.DataFrame({
    'filename_val': filenames_val,
    'mild': mild_val,
    'moderate': moderate_val,
    'severe': severe_val,
    'verysevere': verysevere_val
})


train_datagen = ImageDataGenerator(rotation_range= 10, width_shift_range=0.1, height_shift_range=0.1,brightness_range=[0.5, 1.5],
                                   shear_range=0.1, zoom_range=.1, fill_mode='nearest', rescale=1./255, horizontal_flip=True,
                                   vertical_flip=True)
train_generator = train_datagen.flow_from_dataframe(train_df, directory= path_image, target_size=(224, 224), x_col='filename_train',
                                                    y_col=["mild", "moderate", "severe", "verysevere"], class_mode='raw', batch_size= batch_size)

val_generator = train_datagen.flow_from_dataframe(val_df, directory=path_image, x_col= 'filename_val', y_col=["mild", "moderate", "severe", "verysevere"],
                                                  target_size=(224, 224), class_mode='raw', batch_size= batch_size_val)

#history = Model_train(train_generator, val_generator, epochs, batch_size, batch_size_val, path, checkpoint_filepath)

# load all models
members = load_all_models(n_members, load_model_path)
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, train_generator, val_generator, epochs, path_model, checkpoint_filepath)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, val_generator)
yhat = np.argmax(yhat, axis=1)
acc = accuracy_score(y_val, yhat)
print('Stacked Test Accuracy: %.3f' % acc)

#fig = go.Figure(data=[
 #   go.Line(name='Train_acc', x=history1.epoch, y=history.history['accuracy']),
  #  go.Line(name='Validation_acc', x=history1.epoch, y=history.history['val_accuracy'])
#])
#fig.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", font=dict(family="Courier New, monospace", size=13, color="#7f7f7f"))
#fig