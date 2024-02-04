# Steps to solve Emergency vs Non-Emergency vehicle classification problem using Batch Normalization

'''
<ol>1. Loading the dataset</ol>
<ol>2. Pre-processing the data</ol>
<ol>3. Creating training and validation set</ol>
<ol>4. Defining the model architecture</ol>
<ol> <ol>* Adding batch normalization layer(s) </ol></ol>
<ol>5. Compiling the model</ol>
<ol>6. Training the model</ol>
<ol>7. Evaluating model performance</ol>
'''

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

# create random number generator
seed = 42

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

## 2. Pre-processing the data
# converting 3 dimensional image to 1 dimensional image
X = X.reshape(X.shape[0], 224*224*3)

print(X.shape)
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

print((X_train.shape, y_train.shape), (X_valid.shape, y_valid.shape))

## 4. Defining the model architecture

### <ol>Without Batch Normalization</ol>

# defining the model architecture
model=Sequential()

model.add(InputLayer(input_shape=(224*224*3,)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

## 5. Compiling the model
# defining the adam optimizer and setting the learning rate as 10^-5
adam = Adam(learning_rate=1e-5)

# compiling the model

# defining loss as binary crossentropy
# defining optimizer as Adam
# defining metrics as accuracy

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

## 6. Training the model
# training the model for 50 epochs

model_history = model.fit(X_train, y_train, epochs=50, batch_size=128,validation_data=(X_valid,y_valid))

## 7. Evaluating model performance 

from sklearn.metrics import accuracy_score

# Use the model.predict method to get the predicted probabilities
predicted_probabilities = model.predict(X_valid)

# Convert probabilities to class labels; assume binary classification with a threshold of 0.5
predicted_labels = (predicted_probabilities > 0.5).astype('int32')

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_valid, predicted_labels)
print('Accuracy on validation set:', accuracy * 100, '%')

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

## Defining the model architecture

### <ol>With Batch Normalization</ol>

# importing batch normalization layer
from keras.layers import BatchNormalization
# defining the model architecture and adding batch normalization layers
model=Sequential()

model.add(InputLayer(input_shape=(224*224*3,)))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(units=1, activation='sigmoid'))

model.summary()

## Compiling the model
# compiling the model

# defining loss as binary crossentropy
# defining optimizer as Adam
# defining metrics as accuracy

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

## Training the model
# training the model for 50 epochs using Batch Normalization

model_history = model.fit(X_train, y_train, epochs=200, batch_size=128,validation_data=(X_valid,y_valid))

## 7. Evaluating model performance 

from sklearn.metrics import accuracy_score

# Use the model.predict method to get the predicted probabilities
predicted_probabilities = model.predict(X_valid)

# Convert probabilities to class labels; assume binary classification with a threshold of 0.5
predicted_labels = (predicted_probabilities > 0.5).astype('int32')

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_valid, predicted_labels)
print('Accuracy on validation set:', accuracy * 100, '%')
