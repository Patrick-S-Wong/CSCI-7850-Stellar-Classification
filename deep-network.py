# Imports
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Parameters to change quickly
epochs=101
standardize=False
ablate=False

# Take a look at what devices are available for use...
devices = tf.config.get_visible_devices()
# print(devices)

# Hide devices you are -not- going to use (playing nice with others)
# CPU+GPU0 for this example...
# tf.config.set_visible_devices(devices[0:1]+devices[2:3])

# Check the config again now that some are hidden...
# devices = tf.config.get_visible_devices()

# Check status before...
# tf.config.experimental.get_memory_growth(devices[1])

# Set and verify...
# tf.config.experimental.set_memory_growth(devices[1],True)
# tf.config.experimental.get_memory_growth(devices[1])

# Pick a strategy for dispatch...
strategy = tf.distribute.OneDeviceStrategy('cpu:0') # CPU-only
# strategy = tf.distribute.OneDeviceStrategy('gpu:0') # Hamilton Single GPU
# strategy = tf.distribute.OneDeviceStrategy('gpu:1') # Hamilton Single GPU
# strategy = tf.distribute.MirroredStrategy() # Hamilton - Multi-GPU

with strategy.scope():
    # File that holds the data
    file = "star_classification.csv"

    df = pd.read_csv(file)
    classes = np.array(df['class'])
    classesuniq, classesuniqcounts = np.unique(classes, return_counts=True)
    classeslabel = np.array([x for x in classes])
    classes = np.array([0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in classes], dtype='float64')
    classesuniq = np.array([0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in classesuniq], dtype='float64')

    if ablate is True:
        df.drop(['class', 'obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID'], axis=1, inplace=True)
    else:
        df.drop(['class'], axis=1, inplace=True)
    data = np.array(df, dtype='float64')
    
    # Scaling the data
    if standardize is True:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    else:
        pass

    # Shuffling the data
    shuffle = np.random.permutation(data.shape[0])
    data_shuffled = data[shuffle,:]
    classes_shuffled = classes[shuffle]

    # Model architecture
    layers = 1
    learning = 1
    if standardize is True:
        layers = 2
        learning = 0.01
    else:
        layers = 7
        learning = 0.001

    ff_dim = 32
    x = keras.layers.Input(data.shape[1])
    y = x
    for _ in range(layers):
        i = y
        y = keras.layers.Dense(ff_dim,activation='tanh')(y)
    y = keras.layers.Dense(len(np.unique(classes)),
                        activation=keras.activations.softmax)(y)
    model = keras.Model(x,y)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=learning),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=keras.metrics.SparseCategoricalAccuracy())

    history = model.fit(data,classes,
    validation_split=0.2,
    epochs=epochs,
    verbose=0)

print("Validation accuracy:",*["%.8f"%(x) for x in history.history['val_sparse_categorical_accuracy'][0::10]])
