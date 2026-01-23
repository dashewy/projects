import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras import layers, Input, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping 
from rotten_vis import rotten_eval_visual

data_path_app = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset/Apple'
data_path_straw = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset/Strawberry'
data_path_ban = '/Users/alex/.cache/kagglehub/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images/versions/4/Fruit Freshness Dataset/Fruit Freshness Dataset/Banana'

batch=5

train_set = image_dataset_from_directory(
    data_path_app,
    validation_split=0.2,
    subset='training',
    seed=123,
    batch_size=batch,
    color_mode='rgb',
    label_mode='categorical'
)

validation_set = image_dataset_from_directory(
    data_path_app,
    validation_split=0.2,
    subset='validation',
    seed=123,
    batch_size=batch,
    color_mode='rgb',
    label_mode='categorical'
)

test_set_with_straw = image_dataset_from_directory(
    data_path_straw,
    batch_size=batch,
    labels='inferred',
    label_mode='categorical'
)

test_set_with_ban = image_dataset_from_directory(
    data_path_ban,
    batch_size=batch,
    labels='inferred',
    label_mode='categorical'
)



# clearing old weights through each run
tf.keras.backend.clear_session()

model = Sequential()

# input layer
model.add(Input(shape=(256, 256, 3)))
# rescale_layer
model.add(layers.Rescaling(1./255))

# adding convulational layer
model.add(layers.Conv2D(16, 3, activation='relu'))
# pooling layer to reduce dim
model.add(layers.MaxPooling2D(pool_size=(5, 5), strides=(3, 3), padding='valid'))
# adding dropout layer to help with overfitting (40% of units)
model.add(layers.Dropout(0.5))
# another conv layer
model.add(layers.Conv2D(12, 3, activation='relu'))
# pooling layer
model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(3, 3), padding='valid'))
# drop out layer for overfitting
model.add(layers.Dropout(0.3))
# last conv layer
model.add(layers.Conv2D(8, 3, activation='relu'))
# last pooling layer
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid'))
# flatten layers to go through the output
model.add(layers.Flatten())
# hidden layer
model.add(layers.Dense(16, activation='relu'))
# adding one more drop out layer to help with overfitting (20% of units)
model.add(layers.Dropout(0.2))
# output layer
model.add(layers.Dense(2, activation='softmax'))

# checking model build
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
)

# early_stopping callback to prevent overfitting
es_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1
)


history_appl = model.fit(
    train_set,
    epochs=100,
    validation_data=validation_set,
    callbacks=[es_callback]
)
# early stop pulled out at step 24

model.predict(test_set_with_straw)
model.evaluate(test_set_with_straw)

model.predict(test_set_with_ban)
model.evaluate(test_set_with_ban)

# from visual model is better learning how to predict strawberries
rotten_eval_visual(test_set_with_straw, model)
# from the visual appears model is guessing rotten for every banana (85% accuracy)
rotten_eval_visual(test_set_with_ban, model)