### 1. Importing Libraries and Data preprocessing
# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# set seed
seed = 42

# load csv file
data = pd.read_csv('data/Dataset/emergency_classification.csv')

print(data.head())

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

# creating a training and validation set
X_train, X_valid, y_train, y_valid=train_test_split(X,y,test_size=0.3, random_state=seed)

print("X_train:", X_train.shape, "X_valid:", X_valid.shape)
print("y_train:", y_train.shape, "y_valid:", y_valid.shape)

## Image Augmentation using Keras (ImageDataGenerator)
from keras.preprocessing.image import ImageDataGenerator
image_augmentation = ImageDataGenerator(rotation_range=30, width_shift_range=40, height_shift_range=40, 
                              horizontal_flip=True, vertical_flip=True, fill_mode="nearest")


image_augmentation.fit(X_train)

 ### 2. Model Building
# importing functions from keras to define the neural network architecture
from keras.layers import Dense, InputLayer, Dropout, BatchNormalization, Flatten
from keras.models import Sequential
# importing adam optimizer from keras optimizer module 
from keras.optimizers import Adam

# defining the adam optimizer and setting the learning rate as 10^-5
adam = Adam(learning_rate=1e-5)

# defining and compiling the model architecture
model=Sequential()
model.add(InputLayer(input_shape=(224,224,3)))
model.add(Flatten())
model.add(Dense(100, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(100, activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


# fits the model on batches with real-time data augmentation:
model_history = model.fit_generator(image_augmentation.flow(X_train, y_train, batch_size=128), validation_data=(X_valid, y_valid), epochs=50)

# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()


from sklearn.metrics import accuracy_score

# For binary classification, we assume the output layer uses a sigmoid activation function
# Predicting probabilities for the training set
train_predictions = model.predict(X_train)
# Convert probabilities to binary class labels using a threshold of 0.5
train_predictions_labels = (train_predictions > 0.5).astype(int)

# Calculating training accuracy
training_accuracy = accuracy_score(y_train, train_predictions_labels)
print('Training Accuracy: ', training_accuracy)

# Predicting probabilities for the validation set
valid_predictions = model.predict(X_valid)
# Convert probabilities to binary class labels using a threshold of 0.5
valid_predictions_labels = (valid_predictions > 0.5).astype(int)

# Calculating validation accuracy
validation_accuracy = accuracy_score(y_valid, valid_predictions_labels)
print('\nValidation Accuracy: ', validation_accuracy)

