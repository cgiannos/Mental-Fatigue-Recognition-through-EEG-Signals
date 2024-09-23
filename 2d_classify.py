import numpy as np
import itertools
import pandas as pd
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import is_keras_tensor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import f1_score
from numba import cuda
import gc

# You need no change the paths in lines 48,179,185
# The numpy file raw_mental.npz is not included because of size it is extracted from raw_preprocess.py 
# You need the libraries: numpy, pandas, pathlib, itertools, matplotlib, gc, tensorflow, sklearn, 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # allocate 16GB of memory on the GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=16336)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


batch = 32
epochs = 50
#bands = 5
channels = 63
sampling_points = 769

class_names = ['Fatigued', 'Rested']

data = np.load(r'path')
features = data['features']
features = features.reshape([len(features), channels, sampling_points, 1])
labels = data['labels']
print(features.shape)
print(labels.shape)

def cnn_model():
    model = Sequential()
    inp = layers.Input(shape=(channels, sampling_points, 1))

    conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(channels, sampling_points, 1))(inp) 
    #x22 = layers.Dropout(0.5)(x1)
    conv1 = layers.MaxPooling2D((1, 2), padding='valid')(conv1)
    conv2 = layers.Conv2D(64, ( 3, 3), activation='relu', padding='same')(conv1)
    conv2 = layers.MaxPooling2D((1, 2), padding='valid')(conv2)
    # x444 = layers.Dropout(0.8)(conv2)
    #conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    #conv3 = layers.MaxPooling2D((1, 2), padding='valid')(conv3)


    x13 = layers.Dropout(0.5)(a22)
    x14 = layers.Flatten()(x13)
  
    with tf.device("cpu:0"):
       
        x15 = layers.Dense(512, 'relu')(x14)
    #x16 = layers.Dense(512, 'tanh')(x15)
    out = layers.Dense(2, 'softmax')(x15)
    # assert (keras.backend.is_keras_tensor(inp))
    # assert (keras.backend.is_keras_tensor(out))
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.summary()

    model.compile(Adam(learning_rate=0.001, decay=0.0001), 'categorical_crossentropy', ['accuracy'])
    return model


pat = 5  # this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

all_acc = []

def fit_and_evaluate(t_x, val_x, t_y, val_y, EPOCHS=epochs, BATCH_SIZE=batch):
    model = None
    model = cnn_model()
    history = model.fit(t_x, t_y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping],
                        verbose=1, validation_data=[val_x,val_y])
    _, acc_t = model.evaluate(t_x, t_y)
    print('training accuracy:', str(round(acc_t * 100, 2)) + '%')
    _, acc = model.evaluate(val_x, val_y)
    all_acc.append(acc)
    print('testing accuracy:', str(round(acc * 100, 2)) + '%')
    # print("Val Score: ", model.evaluate(val_x, val_y))
    return history,model



n_folds = 5

# save the model history in a list after fitting so that we can plot later
model_history = []
predicted_targets = np.array([])
actual_targets = np.array([])

kfold = KFold(n_splits=n_folds, shuffle=True, random_state=1)

i = 1
for train_index, val_index in kfold.split(features, labels):

    y = to_categorical(labels, num_classes=2, dtype="int32")

    print("Training on Fold: ", i)

    X_train, X_val = features[train_index], features[val_index]
    y_train, y_val = y[train_index], y[val_index]
    print(len(X_train))
    scaler = None
    scaler = StandardScaler()
    x_train = scaler.fit_transform(X_train.reshape(len(X_train) * sampling_points, channels)).reshape(X_train.shape)
    x_val = scaler.transform(X_val.reshape(len(X_val) * sampling_points , channels)).reshape(X_val.shape)
    
    
#     pca = PCA(n_components=62)
# #     # pca.fit(X_sc_train)
# #     # plt.plot(np.cumsum(pca.explained_variance_ratio_))
# #     # plt.xlabel('Number of components')
# #     # plt.ylabel('Cumulative explained variance')
# #     # plt.show()
#     X_pca_train = pca.fit_transform(X_sc_train)
#     X_pca_test = pca.transform(X_sc_test)
#    # pca_std = np.std(X_pca_train)
#     x_train = X_pca_train.reshape(X_train.shape)
#     x_test = X_pca_test.reshape(X_test.shape)
    print(x_train.shape)
    #print(x_test.shape)
    print(x_val.shape)
    hist,mod = fit_and_evaluate(x_train, x_val, y_train, y_val, epochs, batch)
    model_history.append(hist)
    
    predicted_labels = mod.predict(x_val)
    predicted_labels = np.where(predicted_labels>0.5 , 1, 0)
    predicted_targets = np.append(predicted_targets, tf.argmax(predicted_labels, axis=1))
    actual_targets = np.append(actual_targets, tf.argmax(y_val, axis=1))
    
    #mod.save('C:/Users/giannos/Desktop/seed_preprocessed/models/subject'+str(subj)+'/model'+str(subj)+'_'+str(i)+'.h5')
    i=i+1
    # device = cuda.select_device(0)
    # device.reset()
    gc.collect()

def plot_confusion_matrix(predicted_labels_list, y_val_list):
    cnf_matrix = confusion_matrix(y_val_list, predicted_labels_list)
    np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig(r'/home/st1059685/uploads/biosignals/confusion1.png')
    #plt.show()

# Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.savefig(r'/home/st1059685/uploads/biosignals/confusion2.png')
    #plt.show()

def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


#print(predicted_targets)
plot_confusion_matrix(predicted_targets, actual_targets)

print("Average accuracy : "+ str(round(np.mean(all_acc) * 100, 2)) + "% , std : "+ str(round(np.std(all_acc) * 100, 2)) + "%")
