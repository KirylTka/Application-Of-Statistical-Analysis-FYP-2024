from Functions.import_functions import get_, pp
from Functions.helper_functions import to_dist, plot_dists, helper_cmaps, plot_ims
from Functions.unwrap_functions import uw2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd
from Functions.local_variance_analysis_functions import calc_var
from sklearn.model_selection import train_test_split
import pickle
import os
def gen_data_split(healthy_images,unhealthy_images,test_size,repeats=False):
    '''
    splits data for training
    healthy_images,unhealthy_images: list of images
    test_size: how many images to take from both healthy and hcm for testing
    repeats: how many times to repeat the unhealthy training images to account for the class imbalance


    returns:
        4 lists of training and testing data.
        y data is categorical, meaning the output is [0,1] or [1,0] rather than [0] or [1]

    '''
    if test_size == 0:
        img_h_train = np.array(healthy_images,dtype=object)
        img_uh_train = np.array(unhealthy_images,dtype=object)
        X_train = np.concatenate((img_h_train,img_uh_train))
        y_train = np.concatenate((np.ones((len(img_h_train),1)),np.zeros((len(img_uh_train),1))))
        X_train = np.array([img[:,:,None] for img in X_train],dtype=object)
        y_train = tf.keras.utils.to_categorical(y_train)
        return X_train,[],y_train,[]
    img_h_train, img_h_test = train_test_split(np.array(healthy_images,dtype=object),test_size=test_size,random_state = random.randint(1,1000))
    img_uh_train, img_uh_test = train_test_split(np.array(unhealthy_images,dtype=object),test_size=test_size,random_state = random.randint(1,1000))
    idx_ = ['Healthy','Unhealthy']
    col_ = ['Training','Test']
    info_df = pd.DataFrame([[len(img_h_train),len(img_h_test)],[len(img_uh_train),len(img_uh_test)]],index=idx_,columns = col_)
    #info df just shows how many images are in each set
    if repeats:
        img_uh_train = np.repeat(img_uh_train,repeats)
    X_train = np.concatenate((img_h_train,img_uh_train))
    X_test = np.concatenate((img_h_test,img_uh_test))
    y_train = np.concatenate((np.ones((len(img_h_train),1)),np.zeros((len(img_uh_train),1))))
    y_test = np.concatenate((np.ones((len(img_h_test),1)),np.zeros((len(img_uh_test),1))))
    X_train = np.array([img[:,:,None] for img in X_train],dtype=object)
    X_test = np.array([img[:,:,None] for img in X_test],dtype=object)
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    return X_train,X_test,y_train,y_test


def normalise(healthy_images,unhealthy_images,minn=-90,maxx=90):
    '''
    normalises values to range of -1 to 1 for each image. If angle parameter, dont pass values into minn or maxx.
    If you want it to be automtically selected, set minn,maxx == None

    '''
    imgs_h = []
    imgs_uh = []
    for img in healthy_images:
        img = img.copy()
        if minn is None:
            # img[np.where(img>=4)] = 0 #can be used to remove any anomalous values, MD > 4 is a bit unrealistic and is probably noise.
            minn1 = np.nanmin(img)
            maxx1 = np.nanmax(img)
            mask = np.where(img==0)
            img[np.where(img>=2.5)] = 0
            new_img = 2*(img-minn1)/(maxx1-minn1)-1
            new_img = np.asarray(new_img).astype('float32')
            new_img[np.where(new_img==new_img[0][0])] = 0
            new_img[mask] = 0
            imgs_h.append(new_img)
        else:
            new_img = 2*(img-minn)/(maxx-minn)-1
            new_img = np.asarray(new_img).astype('float32')
            imgs_h.append(new_img)
    for img in unhealthy_images:
        img = img.copy()
        if minn is None:
            # img[np.where(img>=4)] = 0
            minn2 = np.nanmin(img)
            maxx2 = np.nanmax(img)
            mask = np.where(img==0)
            new_img = 2*(img-minn2)/(maxx2-minn2)-1
            new_img = np.asarray(new_img).astype('float32')
            new_img[np.where(new_img==new_img[0][0])] = 0
            new_img[mask] = 0
            imgs_uh.append(new_img)
        else:
            new_img = 2*(img-minn)/(maxx-minn)-1
            new_img = np.asarray(new_img).astype('float32')
            imgs_uh.append(new_img)
    return imgs_h,imgs_uh

def standardise(healthy_images,unhealthy_images):
    ''''
    standardises images using zscore formula
    z = (x-mean)/sd

    '''
    imgs_h = []
    imgs_uh = []

    for img in healthy_images:
        img = img.copy()
        img[np.where(img>=4)] = 0
        mask = np.where(img==0)
        img[mask] = np.nan #set to nan so that its not counted in the mean or var calculation
        new_img = (img-np.nanmean(img))/np.sqrt(np.nanvar(img))
        new_img[np.where(new_img==new_img[0][0])] = np.nan
        new_img[np.isnan(new_img)] = 0
        imgs_h.append(new_img)

    for img in unhealthy_images:
        img = img.copy()
        img[np.where(img>=4)] = 0
        mask = np.where(img==0)
        img[mask] = np.nan
        new_img = (img-np.nanmean(img))/np.sqrt(np.nanvar(img))
        new_img[np.where(new_img==new_img[0][0])] = np.nan
        new_img[np.isnan(new_img)] = 0
        imgs_uh.append(new_img)

    return imgs_h,imgs_uh
def resize_images(X_train,X_test):

    max_height = 64
    max_width = 64
    new_X_train = []
    for image in X_train:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, max_height, max_width)
        padded_image = padded_image.numpy().copy()
        # padded_image[np.where(padded_image==0)] = -1 #this is where you set what you want the region outside of the LV to be
        new_X_train.append(padded_image)

    new_X_test = []
    for image in X_test:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_image = tf.image.pad_to_bounding_box(image, pad_top, pad_left, max_height, max_width)
        padded_image = padded_image.numpy().copy()
        # padded_image[np.where(padded_image==0)] = -1
        new_X_test.append(padded_image)
    return np.array(new_X_train),np.array(new_X_test)


def build_model_cnn(patience,start_from):
    '''
    choose your model architecture here
    '''
    # tf.keras.utils.disable_interactive_logging()
    input_shape = (64,64,1)
    #callback. Stop if the validation loss doesnt reach a new minimum in patience epochs, starting from start_from
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min',start_from_epoch=start_from)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(1, 5, activation='sigmoid', kernel_initializer='he_uniform', input_shape=input_shape,name='Conv2D'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2),name='MaxPooling2D'))
    # model.add(tf.keras.layers.Conv2D(1, 3, activation='sigmoid', kernel_initializer='he_uniform', input_shape=input_shape,name='Conv2D_2'))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2),name='MaxPooling2D_2'))
    model.add(tf.keras.layers.Flatten(name='Flatten'))
    # model.add(tf.keras.layers.Dense(2, activation='sigmoid', kernel_initializer='he_uniform',name='Dense'))
    model.add(tf.keras.layers.Dense(2, activation='softmax',name='Output'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model, callback

    
def train_model(model, callback, data, epochs):
    '''trains model and evalutes'''
    X_train, y_train, X_test, y_test = data

    history = model.fit(X_train,y_train, epochs=epochs, callbacks=[callback],validation_data=(X_test,y_test))
    test_loss, test_acc = model.evaluate(X_test,y_test)
    train_loss, train_acc = model.evaluate(X_train,y_train)
    
    test_loss2 = np.round(test_loss,5)
    train_loss2 = np.round(train_loss,5)
    test_acc2 = 100*test_acc
    train_acc2 = 100*train_acc
    info_df = pd.DataFrame([[test_loss2,test_acc2],[train_loss2,train_acc2]],index=['test','train'], columns=['loss','acc'])
    return history,info_df

    
def plotHistory(history):
    '''
    pass in history from run_model to plot accuracy and loss vs epoch for training and validation
    '''
    history = history.history
    loss = history['loss']
    vloss = history['val_loss']
    acc = [i*100 for i in history['accuracy']]
    vacc = [i*100 for i in history['val_accuracy']]
    X = [i+1 for i in range(len(vacc))]
    
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12,6))
    ax[0].plot(X, acc, '-k',label='Training Accuracy')
    ax[0].plot(X, vacc, '-r', label='Validation Accuracy')
    ax[0].set_title('Accuracy Vs Epoch')
    ax[0].set_xlabel('Epoch',fontsize=18)
    ax[0].set_ylabel('Accuracy (%)',fontsize=18)
    ax[0].tick_params(axis='both', which='major', labelsize=12)
    ax[0].axis('tight')
    ax[0].legend(fontsize=18)
    ax[0].grid('True')
    ax[1].plot(X, loss, '-k', label='Training Loss')
    ax[1].plot(X, vloss, '-r', label='Validation Loss')
    ax[1].set_title('Loss Vs Epoch')
    ax[1].set_xlabel('Epoch',fontsize=18)
    ax[1].set_ylabel('Loss',fontsize=18)
    ax[1].axis('tight')
    ax[1].tick_params(axis='both', which='major', labelsize=12)
    ax[1].legend(fontsize=18)
    ax[1].grid('True')
    plt.show()

def save_model(name,model,data,history):
    '''
    saves the model, training and validation data, and training history
    '''
    if not os.path.exists(f'./Models/{name}'):
        os.makedirs(f'./Models/{name}')
    with open(f'./Models/{name}/{name}_data.pkl', 'wb') as file: 

        pickle.dump(data, file)
    with open(f'./Models/{name}/{name}_history.pkl', 'wb') as file: 
      
        pickle.dump(history, file)
    model.save(f'./Models/{name}/{name}_model.keras')

def load_model(name):
    '''
    Load the model, training and validation data, and training history
    '''
    new_model = tf.keras.models.load_model(f'./Models/{name}/{name}_model.keras')
    with open(f'./Models/{name}/{name}_history.pkl', 'rb') as file: 
      
    # Call load method to deserialze 
        history = pickle.load(file) 

    plotHistory(history)
    with open(f'./Models/{name}/{name}_data.pkl', 'rb') as file: 

        data = pickle.load(file)
    return new_model,data,history
'''
Example Code

param = 'E2A'
health = 'HCM'
method = 'STEAM'
hs = get_('Healthy',method,'SYSTOLE',param,'myo')
hd = get_('Healthy',method,'DIASTOLE',param,'myo')
uhs = get_('HCM',method,'SYSTOLE',param,'myo')
uhd = get_('HCM',method,'DIASTOLE',param,'myo')
hs = pp(hs)
hd = pp(hd)
uhs = pp(uhs)
uhd = pp(uhd)


h_imgs = hd[0] #retrieve only images not maps
uh_imgs = uhd[0] #retrieve only images not maps
X_train,X_test,y_train,y_test = gen_data_split(h_imgs,uh_imgs,5) #create data split, taking 5 from healthy and 5 from unhealthy
X_train,X_test = normalise(X_train,X_test)
# X_train,X_test = standardise(X_train,X_test)
X_train,X_test = resize_images(X_train,X_test)

data = X_train, y_train, X_test, y_test

patience = 50
start_from = 100
epochs = 2000

model, callback = build_model_cnn(patience,start_from)
model.summary()
history,info = run_model(model,callback,data,epochs)
plotHistory(history)

model_weights = new_model.get_weights()
convolution_weights = model_weights[0].squeeze()
fig, ax = plt.subplots()
im = ax.imshow(convolution_weights[:,:,0],**helper_cmaps([convolution_weights]))
cbar = fig.colorbar(im,shrink=0.5,orientation='horizontal')
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Kernel Weights',size=14)
ax.set_axis_off()
ax.tick_params(axis='both', which='major', labelsize=12)
plt.show()


layer_outputs = [layer.output for layer in new_model.layers]
feature_extraction_model = tf.keras.models.Model(inputs=new_model.input,outputs=layer_outputs)
activations = feature_extraction_model.predict(np.array(X_train))
person = 3


feature_map = activations[0][person]




fig, ax = plt.subplots(figsize=(10,6))
ax.set_axis_off()
im= ax.imshow(feature_map,cmap=helper_cmaps([feature_map])['cmap'],vmin=0,vmax=1)
cbar = fig.colorbar(im,shrink=0.5,orientation='horizontal',ticks=[0,0.5,1])
cbar.set_label('Convolutional Layer Activation',size=14)
cbar.ax.tick_params(labelsize=12)
plt.show()

'''


