### Image Augmentation Techniques
'''
1. Image Rotation
2. Image Shifting
3. Image Flipping
4. Image Noising
5. Image Blurring
'''

# importing libraries
import numpy as np
import pandas as pd
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tqdm as tqdm


# libraries for performing image augmentation tasks
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.transform import AffineTransform, warp

### 1. Image Rotation

# reading the csv file
data = pd.read_csv('data/Dataset/emergency_classification.csv')

print(data.head())

# create random number generator
seed = 42

# load images and store it in numpy array

# empty list to store the images
X = []
# iterating over each image
for img_name in data.image_names:
    # loading the image using its name
    img = plt.imread('data/Dataset/images/' + img_name)
    # normalizing the pixel values
    img = img/255
    # saving each image in the list
    X.append(img)
    
# converting the list of images into array
X=np.array(X)

# storing the target variable in separate variable
y = data.emergency_or_not.values

# shape of original dataset
X.shape, y.shape

print(X.shape, y.shape)

# visualizing images
fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
for i in range(5):
    ax[i].imshow(X[i*400])
    ax[i].axis('off')

plt.show()

## 2. Creating training and validation set
# creating a training and validation set
X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.3, random_state=seed)

# shape of training and validation set
(X_train.shape, y_train.shape), (X_valid.shape, y_valid.shape)

print((X_train.shape, y_train.shape), (X_valid.shape, y_valid.shape))

## 3. Augmenting Images
# augmenting the training images
final_train_data = []
final_target_train = []
for i in tqdm.tqdm(range(X_train.shape[0])):
    # original image
    final_train_data.append(X_train[i])
    # image rotation
    final_train_data.append(rotate(X_train[i], angle=30, mode = 'edge'))
    # image flipping (left-to-right)    
    final_train_data.append(np.fliplr(X_train[i]))
    # image flipping (up-down) 
    final_train_data.append(np.flipud(X_train[i]))
    # image noising
    final_train_data.append(random_noise(X_train[i],var=0.2))
    for j in range(5):
        final_target_train.append(y_train[i])

# converting images and target to array
final_train = np.array(final_train_data)
final_target_train = np.array(final_target_train)


# shape of new training set
final_train.shape, final_target_train.shape 

print(final_train.shape, final_target_train.shape )

# visualizing the augmented images
fig,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,20))
for i in range(5):
    ax[i].imshow(final_train[i+30])
    ax[i].axis('off')
plt.show()

# converting 3 dimensional image to 1 dimensional image
final_train = final_train.reshape(final_train.shape[0], 224*224*3)
final_train.shape

print(final_train.shape)

# minimum and maximum pixel values of training images
final_train.min(), final_train.max()

print(final_train.min(), final_train.max())

# converting 3 dimensional validation image to 1 dimensional image
final_valid = X_valid.reshape(X_valid.shape[0], 224*224*3)
final_valid.shape

print(final_valid.shape)

# minimum and maximum pixel values of validation images
print(final_valid.min(), final_valid.max())

## 4. Defining the model architecture
from keras.layers import Dense, InputLayer, Dropout, BatchNormalization
from keras.models import Sequential
# importing adam optimizer from keras optimizer module 
from keras.optimizers import Adam

# defining the adam optimizer and setting the learning rate as 10^-5
adam = Adam(lr=1e-5)

# defining and compiling the model architecture
model=Sequential()

model.add(InputLayer(input_shape=(224*224*3,)))
model.add(Dense(100, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


# summary of the model
model.summary()

# training the model
model_history = model.fit(final_train, final_target_train, epochs=50, batch_size=128,validation_data=(final_valid,y_valid))

from sklearn.metrics import accuracy_score

# Use the model.predict method to get the predicted probabilities
predicted_probabilities = model.predict(final_valid)

# Convert probabilities to binary class labels using a threshold of 0.5
predicted_labels = (predicted_probabilities > 0.5).astype('int32')

## 6. Evaluating model performance
# Calculate and print the accuracy on the validation set
accuracy = accuracy_score(y_valid, predicted_labels)
print('Accuracy on validation set:', accuracy * 100, '%')




