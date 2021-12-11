#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install numpy


# In[2]:


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[3]:


import tensorflow
print(tensorflow.__version__)


# In[4]:


# re-size all the images 
IMAGE_SIZE = [224, 224]

train_path = 'cell_images/Train'
valid_path = 'cell_images/Test'


# In[5]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

mobilnet = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[6]:


mobilnet.summary()


# In[7]:


# don't train existing weights
for layer in mobilnet.layers:
    layer.trainable = False


# In[8]:


# useful for getting number of output classes
folders = glob('cell_images/Train/*')


# In[9]:


folders


# In[10]:


# our layers - you can add more if you want
x = Flatten()(mobilnet.output)


# In[11]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=mobilnet.input, outputs=prediction)


# In[12]:


# view the structure of the model
model.summary()


# In[13]:


from tensorflow.keras.layers import MaxPooling2D


# In[14]:


### Create Model from scratch using CNN
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()


# In[15]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[16]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[17]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('cell_images/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 50,
                                                 class_mode = 'categorical')


# In[18]:


training_set


# In[19]:


test_set = test_datagen.flow_from_directory('cell_images/Test',
                                            target_size = (224, 224),
                                            batch_size = 50,
                                            class_mode = 'categorical')


# In[20]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[21]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[22]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_vgg19.h5')


# In[ ]:





# In[23]:


y_pred = model.predict(test_set)


# In[24]:


y_pred


# In[25]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[26]:


y_pred


# In[ ]:





# In[27]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[28]:


model=load_model('model_vgg19.h5')


# In[ ]:





# In[49]:


img=image.load_img('cell_images/Test/Uninfected/check.png',target_size=(224,224))


# In[50]:


x=image.img_to_array(img)
x


# In[51]:


x.shape


# In[52]:


x=x/255


# In[53]:


x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[54]:


model.predict(img_data)


# In[55]:


a=np.argmax(model.predict(img_data), axis=1)


# In[ ]:





# In[57]:


if(a==1):
    print("Uninfected")
else:
    print("Infected")


# In[ ]:





# In[ ]:




