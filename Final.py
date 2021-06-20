import PIL.Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tkinter import *
import tkinter
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import cv2

# Importing keras and its deep learning tools - neural network model, layers, contraints, optimizers, callbacks and utilities
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import RMSprop, SGD 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l2
from keras.initializers import RandomNormal, VarianceScaling

# Importing scikit-learn tools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#catagory = 'cars' 'football_clubs'
#catagory = 'football_clubs'


cars = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge',
        'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada',
        'Lancia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi',
        'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Saab', 'Seat',
        'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']
football_clubs=['Barcelona', 'Real Madrid', 'Manchester United', 'Borussia Dortmund','Inter Milan', 'Chelsea']
description = ['German','Italy','Egypt','spain']



def ImageConvert(n, i,img_x,img_y):
       im_ex = i.reshape(n, img_x, img_y, 3)
       im_ex = im_ex.astype('float32') / 255
       im_ex = np.subtract(im_ex, 0.5)
       im_ex = np.multiply(im_ex, 2.0)
       return im_ex

def main():
    model = Sequential()
    if catagory == 'cars':
        img_x=50
        img_y=50
    elif catagory == 'football_clubs':
        img_x=224
        img_y=224
    n_channels = 3 
    model.add(Conv2D(32, (3,3),
                     input_shape=(img_x,img_y,n_channels),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, (3,3),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, (3,3),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256, (3,3),
                     padding='valid',
                     bias_initializer='glorot_uniform',
                     kernel_regularizer=l2(0.00004),
                     kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    
    model.add(Dense(4096, activation='relu', bias_initializer='glorot_uniform'))
    model.add(Dropout(0.5))
    
    # final activation is softmax, tuned to the number of classes/labels possible
    
    if catagory == 'cars':
        model.add(Dense(len(cars), activation='softmax'))
    elif catagory == 'football_clubs':
        model.add(Dense(len(football_clubs), activation='softmax'))
    if catagory == 'cars':
        model.load_weights('D:\College\ImageProcessing\projectCode\car.h5py')
    elif catagory == 'football_clubs':
        model.load_weights('D:\College\ImageProcessing\projectCode\images\Clubimages/football.h5py')
    
   
    print(filename+'done')
    im = PIL.Image.open(filename).convert("RGB")
    new_im = np.array(im.resize((img_x,img_y))).flatten()
    filtered_image = cv2.medianBlur(new_im, 7)
    m = int(model.predict_classes(ImageConvert(1, filtered_image,img_x,img_y), verbose=0))
    #txt = 'Made in ' + description[m]
    #plt.text(0, 60,txt , ha='center',fontsize=15,backgroundcolor = 'black',color = 'white',fontstyle =  'oblique')
    plt.imshow(new_im.reshape(img_x, img_y, 3))
    print(m)
    if catagory == 'cars':
        plt.title('Predicted brand: '+cars[m], size=24)
    elif catagory == 'football_clubs':
        plt.title('Predicted brand: '+football_clubs[m], size=24)

    #print(description[m])
        
    plt.show()

def clicked_button1():
    global filename
    filename = askopenfilename()
    
def clicked_button2():
    global catagory
    catagory = combo.get()  # Method taken from user
    main()



window = Tk()
window.title("LOGO DETECTOR")
#window.wm_iconbitmap('f.ico')
window.geometry('650x500')
lbl = Label(window, text="Choose the Log you want to detect",font=("Arial Bold", 10 ),bg="blue",fg="white", pady=5,  height=2, width=30)
combo =ttk.Combobox(window)
lbl2=Label(window,text="choose the category you want to search" )
combo['values']= ("Choose category","football_clubs", "cars")
combo.current(0) #set the selected item

button1 = Button(window, text="Choose the logo to be detected",bg="blue", fg="white", command=clicked_button1)
button2= Button(window, text="search",bg="black", fg="white" , command=clicked_button2)
combo.grid(column=1, row=20 , pady=35)
button1.grid(column=1, row=50 , pady=5)
button2.grid(column=1, row=60,pady=15)
lbl.grid(column=1, row=1)
lbl2.grid(column=0 , row=20)

window.mainloop()

# Importing standard ML set - numpy, pandas, matplotlib

