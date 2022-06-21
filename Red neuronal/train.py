# Main requirements 

from configparser import Interpolation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
import os
#import pandas as pd
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import LabelBinarizer

def myModel(num_output_classes, images_shape):

	# Hyperparameter selection

    # Filters of the CNN

    no_Of_Filters1=30
    no_Of_Filters2=60

    # Shape of the filters used in the CNN
    #size_of_Filter=(5,5)
    size_of_Filter2=(3,3)

    # Tekes batches of 2x2 pixels and avg the

    size_of_pool=(2,2)

    # Nodes of the neural classifier
    
    no_Of_Nodes = 500

    model = Sequential()
    # --------------------------------------------------------------------------------
    # to do: Add layers as presented in the class to conform your CNN
    # ----------------------------- First Convolution -----------------------------
    # Convolution (filtering) with 128 kernels, size of (3, 3):
    model.add(Conv2D(no_Of_Filters2,size_of_Filter2 , data_format="channels_last", input_shape = images_shape))
    model.add(Activation("relu"))
    # ----------------------------- Second Convolution -----------------------------
    # Convolution (filtering) with 128 kernels, size of (3, 3):
    model.add(Conv2D(no_Of_Filters2, size_of_Filter2))
    model.add(Activation("relu"))

    # Max pooling:
    # Max pooled with a kernel of size (2,2)
    model.add(MaxPooling2D(pool_size=size_of_pool))
    # ----------------------------- Third Convolution -----------------------------
    # Convolution (filtering) with 64 kernels, size of (3, 3):
    model.add(Conv2D(no_Of_Filters1,size_of_Filter2))
    model.add(Activation("relu"))
    # ----------------------------- Fourth Convolution -----------------------------
    # Convolution (filtering) with 64 kernels, size of (3, 3):
    model.add(Conv2D(no_Of_Filters1,size_of_Filter2))
    model.add(Activation("relu"))

    # Max pooled with a kernel of size (2,2)
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    # Implement the fully connected layer with N neurons
    # N is a tunable hyper parameter:
    model.add(Dense(no_Of_Nodes))
    model.add(Activation("relu"))
    model.add(Dropout(0.4))
    # Finally, the softmax classifier
    model.add(Dense(num_output_classes))
    model.add(Activation("softmax"))
    # ------------------------------------------------------------------------------

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss=tf.keras.losses.BinaryCrossentropy(),metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.FalseNegatives()])
        

    return model

    ################# Parameters #####################
tf.config.set_visible_devices([], 'GPU')

mainPath = os.path.join("/home/brenda", "cnn2") # principal folder 
dataSetPath = os.path.join(mainPath, "dataset_signals", "train") # folder with all the class folders
outputPath = os.path.join(mainPath, "output_signals") # Folder with results 
#labelFile = 'labels.csv' # file with all names of classes
batch_size_val=64  # how many to process together before updating the interanl parameters
steps_per_epoch_val=100 # we divide all our database in 10 bathces 
epochs_val=55
imageDimesions = (35,35,3)
testRatio = .001    # if 1000 images split will 200 for testing
validationRatio = 0.2 # if 1000 images 20% of remaining 800 will be 160 for validation
###################################################

############################### Importing of the Images
count = 0
images = []
classNo = []
myList = os.listdir(dataSetPath)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")

#Import names
for x in range (len(myList)):
    myPicList = os.listdir(dataSetPath+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(dataSetPath+"/"+str(count)+"/"+y)
        resImg = cv2.resize(curImg, (35, 35), interpolation = cv2.INTER_AREA)
        images.append(resImg)
        classNo.append(count)
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

lb = LabelBinarizer()
labelsEncoded = lb.fit_transform(classNo)
    
X_train, X_test, y_train, y_test = train_test_split(images, labelsEncoded, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
 
# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID

############################### TRAIN
# Create model structure
model = myModel(num_output_classes = len(myList) , images_shape = (35,35,3))
dataGen = ImageDataGenerator()
print(model.summary())
# Train the model
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)
#model.save('saved_model/my_model')
model.save(os.path.join(outputPath, "model"))

############################### PLOT
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')

# Save plot to disk:
plt.savefig(outputPath)
#plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
