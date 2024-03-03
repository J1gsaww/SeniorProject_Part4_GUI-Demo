from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
import numpy as np
from tkinter import Frame
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import os
import random

model_path = "Lastest Model\\best_model3cls.h5"
model = load_model(model_path)

image = None
def upload():
    global image, input_text
    f_types = [('All Files', '*.*')]
    path = filedialog.askopenfilename(filetypes=f_types)

    if path:
        try:
            file_name = os.path.basename(path)
            pil_image = Image.open(path)
            image = np.array(pil_image)

            if image is not None:
                image = cv2.resize(image, (100, 100))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image1 = Image.fromarray(image)
                image1 = ImageTk.PhotoImage(image1)

                #Display the uploaded image in the input frame
                panelA.configure(image=image1)
                panelA.image = image1

                #Move the path label below the axes
                input_text.delete(0, END)
                input_text.insert(0, file_name)
            else:
                print(f"Error: Unable to read the image at {file_name}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Error: No file selected.")
    
#Prediction
target_img_shape =(100,100)
train = 'train'
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)
train_set = train_datagen.flow_from_directory(train,
                                              target_size = target_img_shape,
                                              batch_size = 32,
                                              class_mode='sparse')
labels = (train_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels

def predict():
    global result_image
    global input_image_tk
    global image

    if hasattr(image, 'shape') and len(image.shape) == 3:
        img = cv2.resize(image, (100, 100))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_class = labels[np.argmax(predictions, -1)[0]]

        result_text.delete(0, 'end')
        result_text.insert(INSERT, f"Predicted Class: {predicted_class.encode('utf-8').decode('utf-8')}")

        #Load a random image from the predicted class in the test folder
        test_folder_path = "test"
        class_folder_path = os.path.join(test_folder_path, str(predicted_class))

        #List all files in the class folder
        class_files = os.listdir(class_folder_path)
        random_image_filename = random.choice(class_files)

        random_image_path = os.path.join(class_folder_path, random_image_filename)
        result_path = (random_image_filename)
        result_path_text.delete(0, END)
        result_path_text.insert(0, result_path)

        if class_files:

            #Load the random image and display it in frameB
            random_image = Image.open(random_image_path)
            random_image = random_image.resize((100,100))

            #Check if the image is loaded successfully
            if random_image is not None:
                random_image_tk = ImageTk.PhotoImage(random_image)

                panelB.configure(image=random_image_tk)
                panelB.image = random_image_tk
                #Update frameB with the input image
                #input_image_pil = Image.fromarray(cv2.cvtColor(random_image_tk, cv2.COLOR_BGR2RGB))
                input_image_tk = ImageTk.PhotoImage(random_image_tk)
                panelB.configure(image=input_image_tk)
                panelB.image = input_image_tk

            else:
                result_text.delete(0, 'end')
                result_text.insert(INSERT, "Error: Unable to load random image.")
        else:
            result_text.delete(0, 'end')
            result_text.insert(INSERT, "No Image found in the selected folder")

    else:
        result_text.delete(0, 'end')
        result_text.insert(INSERT, "Please upload an image first.")

def clear_image():
    panelA.configure(image="")
    panelB.configure(image="")
    input_text.delete(0, "end")
    result_text.delete(0, "end")
    result_path_text.delete(0, "end")
    global image
    image = None

def close_program():
    root.destroy()

root = Tk()
root.title("TCHR")

root.configure(bg="#18122B")
root.minsize(800, 600)
root.pack_propagate(True)

for i in range(10):
    root.grid_rowconfigure(i, weight=1)
    root.grid_columnconfigure(i, weight=1)

l1 = Label(root, text="THAI CHARACTERS HANDWRITTEN RECOGNITION",
           fg="white", bg="#443C68", width=30, borderwidth=5, font=('Courier', 16))
l1.grid(row=0, column=1, columnspan=6, padx=20, pady=20, sticky='nsew')

style = ttk.Style()
style.configure('TButton', font=('Arial', 10), padding=6, relief="flat", background="#b5c8c9")

colorbutt = "#D5CEA3"

btn = Button(root, text="Upload Image", fg="black", bg=colorbutt, command=upload, bd=2, relief="raised", font=('Arial', 10))
btn.grid(row=2, column=0, padx=10, pady=10, sticky='nsew')

btn2 = Button(root, text="Predict", fg="black", bg=colorbutt, command=predict, bd=2, relief="raised", font=('Arial', 10))
btn2.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')

btn3 = Button(root, text="Clear", fg="black", bg="#F0CAA3", command=clear_image, bd=2, relief="raised", font=('Arial', 10))
btn3.grid(row=4, column=0, padx=10, pady=10, sticky='nsew')

btn4 = Button(root, text="Exit", fg="black", bg="#D27685", command=close_program, bd=2, relief="raised", font=('Arial', 10))
btn4.grid(row=5, column=0, padx=10, pady=10, sticky='nsew')

#Create frames for displaying images
frameA = Frame(root,width=100, height=100, bg="black", borderwidth=2, relief="flat")
frameA.grid(row=1, column=1, rowspan=5, columnspan=3, padx=10, pady=10, sticky='nsew')

frameB = Frame(root,width=100, height=100, bg="black", borderwidth=2, relief="flat")
frameB.grid(row=1, column=5, rowspan=5, columnspan=3, padx=10, pady=10, sticky='nsew')

#Placeholder for initial images
panelA = Label(frameA, text="Upload Image", bg="Grey", borderwidth=5)
panelA.pack(fill=BOTH, expand=YES)

panelB = Label(frameB, text="Recognition Image", bg="Grey", borderwidth=5)
panelB.pack(fill=BOTH, expand=YES)

#Input text entry
input_text = Entry(root, width=30, bd=2, relief="sunken", font=('Arial', 12))
input_text.grid(row=8, column=1, columnspan=3, padx=10, pady=5, sticky='w')

#Result path text entry
result_path_text = Entry(root, width=30, bd=2, relief="sunken", font=('Arial', 12))
result_path_text.grid(row=8, column=5, columnspan=3, padx=10, pady=5, sticky='w')

#Result text entry
result_text = Entry(root, width=30, bd=2, relief="sunken", font=('Arial', 20))
result_text.grid(row=9, column=3, columnspan=3, padx=10, pady=10, sticky='nsew')

root.mainloop()
