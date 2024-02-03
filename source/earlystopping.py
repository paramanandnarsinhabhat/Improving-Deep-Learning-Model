## 1. Loading the dataset
# import necessary libraries and functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# importing layers from keras
from keras.layers import Dense, InputLayer
from keras.models import Sequential
# importing adam optimizer from keras optimizer module 
from keras.optimizers import Adam

# train_test_split to create training and validation set
from sklearn.model_selection import train_test_split
# accuracy_score to calculate the accuracy of predictions
from sklearn.metrics import accuracy_score

# reading the csv file
data = pd.read_csv('data/Dataset/emergency_classification.csv')

print(data.head())

# create random number generator
seed = 42
rng = np.random.RandomState(seed)

# load images and store it in numpy array
# empty list to store the images
X = []
# iterating over each image
for img_name in data.image_names:
    # loading the image using its name
    img = plt.imread('data/Dataset/images/' + img_name)
    # saving each image in the list
    X.append(img)

# converting the list of images into array
X=np.array(X)

# storing the target variable in separate variable
y = data.emergency_or_not.values


# shape of the images
print(X.shape)

## 2. Pre-processing the data
# converting 3 dimensional image to 1 dimensional image
X = X.reshape(X.shape[0], 224*224*3)
print(X.shape)

# minimum and maximum pixel values of images
X.min(), X.max()

print(X.min(), X.max())

# normalizing the pixel values
X = X / X.max()

# minimum and maximum pixel values of images after normalizing
X.min(), X.max()

print(X.min(), X.max())

## 3. Creating training and validation set
# creating a training and validation set
X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.3, random_state=seed)

# shape of training and validation set
(X_train.shape, y_train.shape), (X_valid.shape, y_valid.shape)


## 4. Defining the model architecture
# defining the model architecture
model=Sequential()

model.add(InputLayer(input_shape=(224*224*3,)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))
from keras.optimizers import Adam
# compiling the model\
# defining the adam optimizer and setting the learning rate as 10^-5
adam = Adam(lr=1e-5)
# Compile the model and specify the optimizer (Adam) correctly
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

## 6. Setting up Early Stopping

# importing module for early stopping
from keras.callbacks import EarlyStopping

# setting up early stopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, mode='min')

## 7. Training the model using Early Stopping
# training the model for 100 epochs
# training the model for 100 epochs

# defining the early stopping as callback using callbacks parameter of model.fit
model_history = model.fit(X_train, y_train, epochs=100, batch_size=128,validation_data=(X_valid,y_valid), callbacks=[early_stopping])
from sklearn.metrics import accuracy_score

# Assuming binary classification
# accuracy on validation set
accuracy = accuracy_score(y_valid, (model.predict(X_valid) > 0.5).astype(int))
print('Accuracy on validation set:', accuracy * 100, '%')

 # summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

## Hyperparameter tuning for Early Stopping (Increasing the patience value)

# defining and compiling the model

# defining the model architecture
model=Sequential()

model.add(InputLayer(input_shape=(224*224*3,)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# increasing the patience and threshold value
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, mode='min')

# training the model for 100 epochs
model_history = model.fit(X_train, y_train, epochs=100, batch_size=128,validation_data=(X_valid,y_valid), callbacks=[early_stopping])


accuracy = accuracy_score(y_valid, (model.predict(X_valid) > 0.5).astype(int))
print('Accuracy on validation set:', accuracy * 100, '%')
# Assuming binary classification
y_pred = model.predict(X_valid)
y_pred_classes = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_valid, y_pred_classes)
print('Accuracy on validation set:', accuracy * 100, '%')
# accuracy on validation set
# accuracy on validation set

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()