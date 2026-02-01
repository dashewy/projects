import sys
import os
import pandas as pd 
import numpy as np 
from sklearn.utils import compute_class_weight
import tensorflow as tf 
from tensorflow.keras import layers, Input, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping 
from rotten_vis import rotten_eval_visual

data_path_app = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset/Apple'
data_path_straw = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset/Strawberry'
data_path_ban = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset/Banana'

batch = 16

def loader(data):
    
    ds_t = image_dataset_from_directory(
        data,
        validation_split=0.2,
        subset='training',
        seed=123,
        batch_size=batch,
        color_mode='rgb',
        label_mode='binary'
    )
    
    ds_v = image_dataset_from_directory(
        data,
        validation_split=0.2,
        subset='validation',
        seed=123,
        batch_size=batch,
        color_mode='rgb',
        label_mode='binary'
    )
    return ds_t, ds_v


app_train, app_val = loader(data_path_app)
ban_train, ban_val = loader(data_path_ban)
straw_train, straw_val = loader(data_path_straw)

train_set = app_train.concatenate(ban_train).concatenate(straw_train)
validation_set = app_val.concatenate(ban_val).concatenate(straw_val)

# balancing class weights
train_labels = np.concatenate([j.numpy() for _, j in train_set], axis=0).flatten()
# print(train_labels)
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(train_labels), 
    y=train_labels
    )
class_weights = dict(enumerate(class_weights))
print(class_weights)

# shuffling 
AUTOTUNE = tf.data.AUTOTUNE

train_set = train_set.shuffle(500).prefetch(buffer_size=AUTOTUNE)
validation_set = validation_set.prefetch(buffer_size=AUTOTUNE)

# clearing old weights through each run
tf.keras.backend.clear_session()

model = Sequential()

# input layer
model.add(Input(shape=(256, 256, 3)))
# rescale_layer
model.add(layers.Rescaling(1./255))

# adding convulational layer
model.add(layers.Conv2D(16, 3, activation='relu'))
# batch normalization
model.add(layers.BatchNormalization())
# pooling layer to reduce dim
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# adding dropout layer to help with overfitting (40% of units)
model.add(layers.Dropout(0.5))
# another conv layer
model.add(layers.Conv2D(12, 3, activation='relu'))
# batch normalization
model.add(layers.BatchNormalization())
# pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# drop out layer for overfitting
model.add(layers.Dropout(0.3))
# last conv layer
model.add(layers.Conv2D(8, 3, activation='relu'))
# last pooling layer
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# flatten layers to go through the output
model.add(layers.Flatten())
# hidden layer
model.add(layers.Dense(16, activation='relu'))
# adding one more drop out layer to help with overfitting (20% of units)
model.add(layers.Dropout(0.2))
# output layer
model.add(layers.Dense(1, activation='sigmoid'))

# checking model build
model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# early_stopping callback to prevent overfitting
es_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1
)


model.fit(
    train_set,
    epochs=100,
    validation_data=validation_set,
    class_weight=class_weights,
    callbacks=[es_callback]
)

model.save('rotten_model.keras')


model.predict(validation_set)
model.evaluate(validation_set)

rotten_eval_visual(validation_set, app_val.class_names , model)