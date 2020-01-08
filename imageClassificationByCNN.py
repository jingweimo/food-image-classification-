# -*- coding: utf-8 -*-
"""
Classification on Small-scale Image Data based on Data Expansion/Augmentation
Created on Wed Nov 09 22:25:36 2016

Edit history:
  11/20/2016: add early stopping and modelcheckpoint  
  11/22/2016: data splitting of 80% training and 20% testing
  Three training cases: 
      1) 100 epochs
      2) 200 epochs
      3) 400 epochs
      4) 600 epochs
  Obtained best accuracies
      1): 83.37% for training and 87.24% for testing
      2): 87.24% for training and 89.21% for testing
      3): 90.34% for training and 90.41% for testing
      4): 93.08% for training and 89.98% for testing
  
"""

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')
#from PIL import Image
#img = Image.open('Test/apple/4.jpeg')
img = load_img('imagePath')  # this is a PIL image
img.save('saveImagePath')
plt.imshow(img)
x = img_to_array(img)  # this is a Numpy array with shape (3, 128, 128)
x = x.reshape((1,) + x.shape) 

###############################################################################
#                            Prepare data sets                                #
###############################################################################
#Split images into training and testing parts
#Note: run only one time!!
def splitImageSet(rootFolder, outFolder, p):
    import os
    from os import listdir
    import numpy as np
    from PIL import Image
    cats = listdir(rootFolder) # a list of subfolder names    
    for cat in cats:
        print 'Image Category...{}'.format(cat)
        
        folderPath = (os.path.join(rootFolder,cat))
        imgNames = listdir(folderPath)
        imgPaths = [os.path.join(folderPath,imgName) for imgName in imgNames]
        idx = np.random.permutation(len(imgPaths))
        trainIdx = idx[:int(p*len(idx))]
        testIdx = [ind for ind in idx if not ind in trainIdx]

        if not os.path.exists(os.path.join(outFolder,'Train',cat)):
            os.makedirs(os.path.join(outFolder,'Train',cat))
        for k in range(len(trainIdx)):
            img = Image.open(os.path.join(imgPaths[trainIdx[k]]))
            #temp = os.path.join(outFolder,'train',cat,imgNames[trainIdx[k]])
            img.save(os.path.join(outFolder,'Train',cat,imgNames[trainIdx[k]]))            
        if not os.path.exists(os.path.join(outFolder,'Test',cat)):
            os.makedirs(os.path.join(outFolder,'Test',cat))
        for k in range(len(testIdx)):
            img = Image.open(os.path.join(imgPaths[testIdx[k]]))
            img.save(os.path.join(outFolder,'Test',cat,imgNames[testIdx[k]])) 
            
    print 'Split Done!'
    return
rootFolder = 'rootFolder' #add the image directory
outFolder = 'outFolder' #image output directory    
splitImageSet(rootFolder, outFolder, 0.80)    

#Please start from here!!
###############################################################################
#                           Build a CNN model                                 #
###############################################################################
#CNN model
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras import backend as K

model = Sequential()
model.add(Convolution2D(32, 7, 7, input_shape=(3, 128, 128)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #initial lr = 0.01
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
print model.summary()

####################################
#         Callback Schedule       #
###################################
import keras
class decaylr_loss(keras.callbacks.Callback):
    def __init__(self):
        super(decaylr_loss, self).__init__()
    def on_epoch_end(self,epoch,logs={}):
        #loss=logs.items()[1][1] #get loss
        loss=logs.get('loss')
        print "loss: ",loss
        old_lr = 0.001 #needs some adjustments
        new_lr= old_lr*np.exp(loss) #lr*exp(loss)
        print "New learning rate: ", new_lr
        K.set_value(self.model.optimizer.lr, new_lr)
lrate = decaylr_loss()
#early stopping
patience = 20
earlystopper = EarlyStopping(monitor='val_acc', patience=patience, 
                             verbose=1, mode='max')       
#check point
wdir = 'wdir' #work directory
filepath = os.path.join(wdir,'modelWeights','cnnModelDEp80weights.best.hdf5') #save model weights
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

###############################################################################
#                     Data Expansion or Augmentation                          #
###############################################################################
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                                   #featurewise_center=True,
                                   #featurewise_std_normalization=True,
                                   rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

trainDir = 'Data\Train'
train_generator = train_datagen.flow_from_directory(trainDir,  
                                                    target_size=(128,128),
                                                    batch_size=32,
                                                    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)  
testDir = 'Data\Test'
test_generator = test_datagen.flow_from_directory(testDir,
                                                  target_size=(128,128),
                                                  batch_size=32,
                                                  shuffle=False,
                                                  class_mode='categorical')

###############################################################################
##                      Fit, Evaluate and Save Model                          #                                  
###############################################################################
epochs = 100
#epochs = 200
#epochs = 400
#epochs = 600
samples_per_epoch = 4654 
val_samples = 1168

#Fit the model
hist = History()
model.fit_generator(train_generator,
                    samples_per_epoch= samples_per_epoch,
                    nb_epoch=epochs,
                    verbose=1,
                    validation_data=test_generator,
                    nb_val_samples=val_samples, 
                    callbacks = [earlystopper, lrate, checkpoint, hist])

#evaluate the model
scores = model.evaluate_generator(test_generator, val_samples=val_samples) 
print("Accuracy = ", scores[1])

#save model
savePath = wdir
model.save_weights(os.path.join(savePath,'cnnModelDEp80.h5')) # save weights after training or during training
model.save(os.path.join(savePath,'cnnModelDEp80.h5')) #save complied model

#plot acc and loss vs epochs
import matplotlib.pyplot as plt
print(hist.history.keys())
#accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig(os.path.join(savePath,'cmdeP80AccVsEpoch.jpeg'), dpi=1000, bbox_inches='tight')
plt.show()
#loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig(os.path.join(savePath,'cmdeP80LossVsEpoch.jpeg'), dpi=1000, bbox_inches='tight')
plt.show()

###############################################################################                          
#Note: train 4364 images (80%) and test 1458 images (20%)                     # 
# 100 epochs:                                                                 #
# 200 epoches: acc = 0.8724; val_acc = 0.89212                                #  
# 400 epoches:                                                                #
# 600 epochs:                                                                 #
###############################################################################

# load the model
# not necessary the best at the end of training 
from keras.models import load_model
myModel = load_model(os.path.join(savePath,'cnnModelDEp80.h5'))                                  
scores = myModel.evaluate_generator(test_generator,val_samples)
print("Accuracy = ", scores[1])

########################
# Check-pointed model  #
#######################
model = Sequential()
model.add(Convolution2D(32, 7, 7, input_shape=(3, 128, 128)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))
#lr = 0.00277615583366 #adjust lr based on training process
#lr = 0.00181503843077
#lr = 0.00163685841542 #case 2
lr =  0.00122869861281 # case3
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.load_weights(filepath) #load saved weights
scores = model.evaluate_generator(test_generator,val_samples)
print("Accuracy = ", scores[1])


#Confusion matrix on the test images
#imgDir = testDir
imgDir = trainDir
test_generator = test_datagen.flow_from_directory(imgDir,
                                                  target_size=(128,128),
                                                  batch_size=32,
                                                  shuffle=False, 
                                                  class_mode='categorical')
#val_samples = 1168
val_samples = 4654
predict = model.predict_generator(test_generator,val_samples)

yTrue = test_generator.classes
yTrueIdx = test_generator.class_indices

from sklearn.metrics import classification_report, confusion_matrix
yHat = np.ones(predict.shape[0],dtype = int)
for i in range(predict.shape[0]):
    temp = predict[i,:]
    yHat[i] = np.argmax(temp)  
    
from sklearn.metrics import accuracy_score
acc = accuracy_score(yTrue,yHat)
print "Accuracy on test images:", acc #same as scores[1]

def numToLabels(y,cat):
    numLabel = []
    import numpy as np
    yNew = np.unique(y) #sorted 
    for i in range(len(y)):
        idx = np.where(yNew == y[i])[0][0]
        numLabel.append(cat[idx])
    return numLabel                   
#labels = sorted(yTrueIdx.keys())
labels = ['Ap','Ba','Br','Bu','Eg','Fr','Hd','Pz','Rc','St']
yActLabels = numToLabels(yTrue,labels)
yHatLabels = numToLabels(yHat,labels)
CM = confusion_matrix(yActLabels,yHatLabels,labels) #np.array
#print CM    
print(classification_report(yTrue,yHat,target_names=labels))

#Alternatively: pd.crosstab
import pandas as pd
#preds = pd.DataFrame(predict)
y1 = pd.Categorical(yActLabels,categories=labels)
y2 = pd.Categorical(yHatLabels,categories=labels)
pd.crosstab(y1,y2,rownames=['True'], colnames=['Predicted'])

###############################################################################
#                                Miscellaneous                                #
###############################################################################  
#evaluate execution efficiency
import time
t = time.time()
s = model.predict_generator(test_generator,val_samples)
elapsedTime = time.time()-t
print 'Average time: {} second'.format(elapsedTime/val_samples)
#average time: 0.00895 s, less than 0.01 s
def getImgPaths2(rootPath): 
    import os
    from os import listdir
    print 'Extract paths and labels...'
    cats = listdir(rootPath)
    imgPaths = []
    imgLabels = []
    for cat in cats:
        print '{}...'.format(cat)
        foldPaths = os.path.join(rootPath, cat)
        imgPaths.extend([os.path.join(foldPaths,imgName) for imgName in listdir(foldPaths)])
        imgLabels.extend([cat]*len(listdir(foldPaths)))
    return imgPaths, imgLabels 
def getImgData(imgPaths):
    from scipy import misc 
    import numpy as np
    print('Extract image data...')
    temp1 = misc.imread(imgPaths[0])
    imgData = np.zeros((len(imgPaths),temp1.shape[0],temp1.shape[1],temp1.shape[2]),
                       dtype='float32')
    for ii in range(len(imgPaths)):
        temp = misc.imread(imgPaths[ii])
        imgData[ii,:,:,:] = temp
        print "\r{} Percent complete\r".format(100*(ii+1)/len(imgPaths)),
    return imgData
imgPaths = getImgPaths2(trainDir)    
# Expanded image data
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow()
plt.imshow()

#######################################################
#   Images with correct or wroning predicted Labels   #
####################################################### 
imgPaths, imgLabels = getImgPaths2(testDir)
X_test = getImgData(imgPaths) #(1168, 128, 128, 3)
X_test /= 255
test_wrong = [im for im in zip(X_test, yHatLabels, yActLabels) if im[1] != im[2]]
print(len(test_wrong))
#112 misclassified images

#show some misclassified images
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 8))
import numpy as np
for ind, val in enumerate(test_wrong):
    #print ind
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.subplot(7, 8, ind + 1)
    img = val[0]
    img *= 255
    plt.axis("off")
    plt.text(0,0,val[2], fontsize=12, color='blue')
    plt.text(40,0,val[1], fontsize=12, color='red')
    plt.imshow(img.astype('uint8'))
    if ind==55:
        break     
plt.savefig(os.path.join(savePath,'MissClassifiedImages1.jpeg'), 
            dpi=800, bbox_inches='tight')
plt.show()

#show some misclassified images
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 8))
import numpy as np
for ind, val in enumerate(test_wrong[56:]):
    #print ind
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.subplot(7, 8, ind + 1)
    img = val[0]
    img *= 255
    plt.axis("off")
    plt.text(0,0,val[2], fontsize=12, color='blue')
    plt.text(40,0,val[1], fontsize=12, color='red')
    plt.imshow(img.astype('uint8'))
    if ind==55:
        break     
plt.savefig(os.path.join(savePath,'MissClassifiedImages2.jpeg'), 
            dpi=800, bbox_inches='tight')
plt.show()

########################################
#   Rotated, shifted, sheared Images   #
########################################
imgData = getImgData(imgPaths[:9]) #extract 9 images, (9L, 128L, 128L, 3L)
# plot raw images
for i in range(0, 9):
    from scipy import misc
    img = misc.imread(imgPaths[i])
    plt.subplot(3,3,i+1)
    fig = plt.imshow(img)
    ax = plt.gca()
    #ax.set_axis_off()
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
plt.show()
#or
for i in range(0, 9):
    img = imgData[i,:,:,:].astype('uint8')
    plt.subplot(3,3,i+1)
    fig = plt.imshow(img)
    ax = plt.gca()
    ax.set_axis_off()
plt.savefig(os.path.join(savePath,'RawImages.jpeg'), dpi=1000, bbox_inches='tight')
plt.show()

#image distortion by ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator #, array_to_img, img_to_array
datagen = ImageDataGenerator(
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True
                             )
#single fruit
#from PIL import Image
x = imgData[1].reshape((1,)+imgData[0].shape)
fig = plt.imshow(imgData[1].astype('uint8')) #imshow: img_dim_ordering (w,h,channels)
plt.gca().set_axis_off()
plt.savefig(os.path.join(savePath,'rawImage.jpeg'), 
            dpi=1000, bbox_inches='tight')
x = x.transpose(0,3,1,2) #reshaped into 4d for datagen.flow 
i = 0
for x_batch in datagen.flow(x, batch_size=1):
    i += 1
    plt.subplot(3,3,i)
    I = x_batch[0]
    img = (I.transpose(1,2,0)).astype('uint8')
    fig=plt.imshow(img,cmap=plt.get_cmap('gray'))
    plt.gca().set_axis_off()
    if i==9:
        break
plt.savefig(os.path.join(savePath,'expandedImages1.jpeg'), 
            dpi=1000, bbox_inches='tight')
#multiple fruit
for x_batch in datagen.flow(imgData.transpose(0,3,1,2), batch_size=9):
    for i in range(0,9):
        plt.subplot(3,3,i+1)
        I = x_batch[i]
        #img = array_to_img(I)
        img = (I.transpose(1,2,0)).astype('uint8')
        fig = plt.imshow(img,cmap=plt.get_cmap('gray'))
        plt.gca().set_axis_off()
    plt.show()
    break
plt.savefig(os.path.join(savePath,'expandedImages2.jpeg'), 
            dpi=1000, bbox_inches='tight')
  
#evaluate execution efficiency
import time
t = time.time()
s = model.predict_generator(test_generator,val_samples)
elapsedTime = time.time()-t
print 'Average time: {} second'.format(elapsedTime/val_samples)
#average time: 0.00895 s, less than 0.01 s


