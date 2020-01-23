import tensorflow as tf
from tensorflow import keras
import os
import matplotlib
import numpy as np

train_horse_dir = os.path.join('D:/horse-or-human/train/horses')
train_human_dir = os.path.join('D:/horse-or-human/train/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
print(train_horse_names[:5])
print(train_human_names[:5])

print('total horse image in training set:',len(train_horse_names))
print('total human images in training set:',len(train_human_names))


#defining the model
model = tf.keras.Sequential([
    #our first convolution
    tf.keras.layers.Conv2D(16,(3,3),activation = 'relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    #our second convolution
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #our third convolution
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #our forth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    #flatten the result before fitting into neural network
    tf.keras.layers.Flatten(),
    #512 neuron hidden layer
    tf.keras.layers.Dense(512,activation='relu'),
    #only 1 output neuron.0 for horse class and 1 for human class
    tf.keras.layers.Dense(1,activation = 'sigmoid')
])
#see the summary of our model
model.summary()

#compiling the model
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

#training the model from generators
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    'D:/horse-or-human/train',
    target_size = (300,300),
    batch_size= 32,
    class_mode = 'binary'
)

#do the training
#lets train for 50 epochs

model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_generator),
    epochs = 50,
    verbose = 1
)
#test the model(predict)
from keras.preprocessing import image
img_predict = image.load_img('D:/horse-or-human/validation/horses/horse1-122.png')
img_predict = image.img_to_array(img_predict)
img_predict = np.expand_dims(img_predict,axis = 0)

rslt = model.predict(img_predict)
print(rslt)
#if rslt >= 0.5 that is human
#if rslt <0.5 that is horse
if(rslt >= 0.5):
    print("human")
else:
    print("horse")