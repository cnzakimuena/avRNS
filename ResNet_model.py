
# coding: utf-8

# # Residual Networks

# load required packages

# import numpy as np
# from tensorflow.keras import layers
# import tensorflow
import spec_gen
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# import pydot
# from IPython.display import SVG
# from tensorflow.keras.utils import model_to_dot
# from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model
from resnets_utils import *
from tensorflow.keras.initializers import glorot_uniform
# import scipy.misc
# from matplotlib.pyplot import imshow
# get_ipython().magic('matplotlib inline')
import tensorflow.keras.backend as K
import os
from tensorflow.keras.optimizers import Adam
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
os.environ["PATH"] += os.pathsep + 'C:/Users/cnzak/Documents/Graphviz/bin/'

# ## 2 - Building a Residual Network

# ### 2.1 - The identity block

# The identity block is the standard block used in ResNets, and corresponds to the case where the input activation
# has the same dimension as the output activation.


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# ## 2.2 - The convolutional block

# The ResNet "convolutional block" is the second block type. You can use this type of block when the input and output
# dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut
# path.


def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# ## 3 - Building ResNet model (50 layers)

# Leverage the blocks to build a deep ResNet.

# The details of this ResNet-50 model are:
# - Zero-padding pads the input with a pad of (3,3)
# - Stage 1:
#     - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
#     - BatchNorm is applied to the 'channels' axis of the input.
#     - MaxPooling uses a (3,3) window and a (2,2) stride.
# - Stage 2:
#     - The convolutional block uses three sets of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
#     - The 2 identity blocks use three sets of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
# - Stage 3:
#     - The convolutional block uses three sets of filters of size [128,128,512], "f" is 3, "s" is 2 and
#     the block is "a".
#     - The 3 identity blocks use three sets of filters of size [128,128,512], "f" is 3 and
#     the blocks are "b", "c" and "d".
# - Stage 4:
#     - The convolutional block uses three sets of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and
#     the block is "a".
#     - The 5 identity blocks use three sets of filters of size [256, 256, 1024], "f" is 3 and
#     the blocks are "b", "c", "d", "e" and "f".
# - Stage 5:
#     - The convolutional block uses three sets of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and
#     the block is "a".
#     - The 2 identity blocks use three sets of filters of size [512, 512, 2048], "f" is 3 and
#     the blocks are "b" and "c".
# - The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
# - The 'flatten' layer doesn't have any hyperparameters or name.
# - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name
# should be `'fc' + str(classes)`.


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)  # axis=3, corresponds to index which represents the channels
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters = [128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    # X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(1, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


# Run the following code to build the model's graph. If your implementation is not correct you will know it by checking
# your accuracy when running `model.fit(...)` below.

# model = ResNet50(input_shape=(64, 64, 3), classes=6)
model = ResNet50(input_shape=(64, 64, 2), classes=2)  # adapted to project dataset

# As seen in the Keras Tutorial Notebook, prior training a model, must configure the learning process by compiling
# the model.

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
opt = Adam(learning_rate=0.00001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Model is now ready to be trained. The only thing you need is a dataset.

# # Let's load the SIGNS Dataset.
# X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# # image_a = X_train_orig[0, :, :, :]  # to view the first RGB image in 'X_train_orig" tensor
# # import matplotlib.pyplot as plt
# # plt.imshow(image_a)

# Load spectrogram images dataset.
subjects_G1 = "AMD"
subjects_G2 = "normal"
str_dataPath = r'C:/Users/cnzak/Desktop/data/avRNS/biophotonics'
# ML input data : 'drusenConverter' --> [stateFull_Data]; 'ccConverter' --> [stateFull_Data2]
sampling_rate = 200  # sampling frequency, 600/3 = 200 px/mm
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = spec_gen.load_split_spec_dataset(subjects_G1,
                                                                                                 subjects_G2,
                                                                                                 str_dataPath,
                                                                                                 sampling_rate)
# X_orig, Y_orig, classes = spec_gen.load_spec_dataset(subjects_G1, subjects_G2, str_dataPath, sampling_rate)

# # Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.
# X = X_orig/255.

# # Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
# # Convert training and test labels to one hot matrices
# Y_train = convert_to_one_hot(Y_train_orig, 2).T
# Y_test = convert_to_one_hot(Y_test_orig, 2).T

# Y = convert_to_one_hot(Y_orig, 2).T
# Y = Y.astype(np.float32)

# # For input using 'sigmoid' end layer
# Y = Y_orig[0]
# Y = Y.astype(np.int32)

Y_train = Y_train_orig[0]
Y_train = Y_train.astype(np.int32)
Y_test = Y_test_orig[0]
Y_test = Y_test.astype(np.int32)

# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))

# print("number of examples = " + str(X.shape[0]))
# print("X shape: " + str(X.shape))
# print("Y shape: " + str(Y.shape))

# Run the following cell to train your model on 2 epochs with a batch size of 32. On a CPU it should take you around
# 5min per epoch.
# history = model.fit(X, Y, validation_split=0.33, epochs=4, batch_size=32)
history = model.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_test, Y_test),
)

# Let's see how this model (trained on only two epochs) performs on the test set.

# preds = model.evaluate(X_test, Y_test)
# print("Loss = " + str(preds[0]))
# print("Test Accuracy = " + str(preds[1]))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Currently model trains for just two epochs. You can see that it achieves poor performances.

# You can also optionally train the ResNet for more iterations, if you want. Will get a lot better performance when
# trained for ~20 epochs, but this will take more than an hour when training on a CPU.

# Using a GPU, we've trained our own ResNet50 model's weights on the SIGNS dataset. You can load and run our trained
# model on the test set in the cells below. It may take ≈1min to load the model.

# ## 4 - Test on your own image (Optional/Ungraded)

# Can also take a picture of your own hand and see the output of the model. To do this:
#     3. Write your image's name in the following code
#     4. Run the code and check if the algorithm is right! 

# img_path = 'images/my_image.jpg'
# img = image.load_img(img_path, target_size=(64, 64))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = x/255.0
# print('Input image shape:', x.shape)
# my_image = scipy.misc.imread(img_path)
# imshow(my_image)
# print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
# print(model.predict(x))

x = X_test[200, :, :, :]/255
x = np.expand_dims(x, axis=0)  # add dimension to array for concatenation
predictions = model.predict(x)
predicted_class = np.argmax(predictions, axis=-1)

# Can also print a summary of your model by running the following code.

model.summary()

# Run the code below to visualize your ResNet50. You can also download a .png picture of your model

plot_model(model, to_file='model.png')

# ## Reminders
# - Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.  
# - The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to
# learn an identity function.
# - There are two main types of blocks: The identity block and the convolutional block. 
# - Very deep Residual Networks are built by stacking these blocks together.

# ### References 
# 
# This notebook presents the ResNet algorithm due to He et al. (2015). The implementation here also took significant
# inspiration and follows the structure given in the GitHub repository of Francois Chollet:
# 
# - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun -
# [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
# - Francois Chollet's GitHub repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# 
