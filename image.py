import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

from keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers.legacy import Adam

#REMOVE DOGGY IMAGES

data_dir = "/Users/agniswariguha/Img_classification_Agni/data"
os.listdir(os.path.join(data_dir,'happy'))

image_exts = ['jpeg', 'jpg', 'bmp', 'png']  # check img
for image_class in os.listdir(data_dir):

    for image in os.listdir(os.path.join(data_dir, image_class)):
        print(image)

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list{}'.format(image_path))
                os.remove(image_path)
        except exception as e:
            print('Issue with image{}'.format(image_path))
#LOAD DATA

var = tf.data.Dataset
tf.keras.utils.image_dataset_from_directory('data')
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# class 0 = happy class 1=sad
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

scaled = batch[0] / 255

#2.PREPROCESS DATA
#A.SCALE DATA
data = data.map(lambda x, y: (x / 255, y))
scaled_iterator = data.as_numpy_iterator()

batch = scaled_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img)
    ax[idx].title.set_text(batch[1][idx])

#SPLIT DATA
train_size = int(len(data) * .7)
val_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


#DEEP MODEL
#A.BUILD DEEP LEARNING MODEL
model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())  # here 16 filter (3pix * 3pix filter pixel size) and per time there will be 1 pixel

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

#B.TRAIN
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
print(hist.history)

#PLOT PERFORMANCE
#PLOT LOSS
fig = plt.figure()
plt.plot(hist.history['loss'],color='teal',label='loss')
plt.plot(hist.history['val_loss'],color='orange',label='val_loss')
fig.suptitle('Loss',fontsize=20)
plt.legend(loc="upper left")
print(plt.show())

#plot ACCURACY
fig = plt.figure()
plt.plot(hist.history['accuracy'],color='teal',label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=20)
plt.legend(loc="upper left")
print(plt.show())

#EVALUATE PERFORMANCE

#EVALUATE
from tensorflow.keras.metrics import Precision,Recall,BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
print(len(test))

for batch in test.as_numpy_iterator():
    X,y = batch
    yhat = model.predict(X)
    pre.update_state(y,yhat)
    re.update_state(y,yhat)
    acc.update_state(y,yhat)

print(f'Precision:{pre.result().numpy()},Recall:{re.result().numpy()},Accuracy:{acc.result().numpy()}')

#TEST

img = cv2.VideoCapture(0) #cv2.imread('happyman.jpeg')
plt.imshow(img)
print(plt.show())

resize = tf.image.resize(img,(256,256))
plt.imshow(resize.numpy().astype(int))
print(plt.show())

yhat = model.predict(np.expand_dims(resize/255,0))
print(yhat)

if yhat > 0.5:
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')

#SAVE THE MODEL
#from tensorflow.keras.models import load_model
#model.save(os.path.join('models','happysadmodel.h5'))

#new_model = load_model(os.path.join('models','happysadmodel.h5'))
#yhat_new=new_model.predict(np.expand_dims(resize/255,0))

#if yhat > 0.5:
 #   print(f'Predicted class is Sad')
#else:
  #  print(f'Predicted class is Happy')