import PIL.Image
from PIL import ImageTk

from tkinter import *
import tkinter
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Importing keras and its deep learning tools - neural network model, layers, contraints, optimizers, callbacks and utilities
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import  RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l2
from keras.initializers import RandomNormal, VarianceScaling
import natsort
# Importing scikit-learn tools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix





cars = ['Alfa Romeo', 'Audi', 'BMW', 'Chevrolet', 'Citroen', 'Dacia', 'Daewoo', 'Dodge',
        'Ferrari', 'Fiat', 'Ford', 'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Lada',
        'Lancia', 'Land Rover', 'Lexus', 'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi',
        'Nissan', 'Opel', 'Peugeot', 'Porsche', 'Renault', 'Rover', 'Saab', 'Seat',
        'Skoda', 'Subaru', 'Suzuki', 'Tata', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo']
football_clubs=['Barcelona', 'Real Madrid', 'Manchester United', 'Borussia Dortmund','Inter Milan', 'Chelsea']




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
        model.load_weights('car.h5py')
    elif catagory == 'football_clubs':
        model.load_weights('football.h5py')
    
   
    
    im = PIL.Image.open(filename).convert("RGB")
    new_im = np.array(im.resize((img_x,img_y))).flatten()
    filtered_image = cv2.medianBlur(new_im, 7)
    m = int(model.predict_classes(ImageConvert(1, filtered_image,img_x,img_y), verbose=0))

    if catagory == 'cars':
        root = tkinter.Toplevel()
        root.title("OUTPUT")
        root.geometry('500x500')
        canvas = Canvas(root, width = 300, height = 500)
        canvas.pack()
        image = PIL.Image.open(filename)
        img = ImageTk.PhotoImage(image)
        width, height = image.size
        print (width)
        print(height)
        im=img
        if width < 100 :
           im = img._PhotoImage__photo.zoom(3)
        

        canvas.create_image(150, 150, anchor=NW, image=im)
        name=StringVar()
        lbl3=Label(canvas,textvariable=name,font=("Arial Bold", 10 ),fg="black")
        lbl4=Label(canvas,text="The Predicted brand:",font=("Arial Bold", 10 ),fg="black")
        lbl3.place(x=140, y=400)
        lbl4.place(x=0, y=400)
        name.set(cars[m])
        root.mainloop()

    elif catagory == 'football_clubs':
        root = tkinter.Toplevel()
        root.title("OUTPUT")
        root.geometry('500x500')
        canvas = Canvas(root, width = 300, height = 500)
        canvas.pack()
        img = ImageTk.PhotoImage(PIL.Image.open(filename))
        canvas.create_image(20, 20, anchor=NW, image=img)
        name=StringVar()
        lbl3=Label(canvas,textvariable=name,font=("Arial Bold", 10 ),fg="black")
        lbl4=Label(canvas,text="The Predicted brand:",font=("Arial Bold", 10 ),fg="black")
        lbl3.place(x=140, y=400)
        lbl4.place(x=0, y=400)
        name.set(football_clubs[m])
        root.mainloop()
                

def clicked_button1():
    global filename
    filename = askopenfilename()
    
def clicked_button2():
    global catagory
    catagory = combo.get()  # Method taken from user
    main()




window = Tk()
window.title("LOGO DETECTOR")
window.geometry('650x500')
lbl = Label(window, text="Choose the Logo you want to detect",font=("Arial Bold", 10 ),bg="blue",fg="white", pady=5,  height=2, width=30)
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

