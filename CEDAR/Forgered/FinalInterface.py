import cv2
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter.filedialog as fd
import tkinter.font as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import seaborn as sns
from skimage import exposure
from skimage import feature
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.classifier import ClassificationReport


from PIL import Image
from PIL import ImageTk
from skimage import exposure
from skimage import feature
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import glob
import tkinter
from tkinter import *
from PIL import Image, ImageTk
import tkinter as Tk
from tkinter.ttk import Progressbar
import tkinter.ttk as ttk


class DemoSplashScreen:
    def __init__(self, parent):
        self.parent = parent


        self.aturSplash()
        self.aturWindow()

    def aturSplash(self):
        # import image menggunakan Pillow
        self.gambar = Image.open('sigg.png')
        self. gambar= self.gambar.resize((800, 450), Image.ANTIALIAS)
        self.imgSplash = ImageTk.PhotoImage(self.gambar)


    def aturWindow(self):
        # ambil ukuran dari file image
        lebar, tinggi = self.gambar.size


        setengahLebar = (self.parent.winfo_screenwidth() - lebar) // 2
        setengahTinggi = (self.parent.winfo_screenheight() - tinggi) // 2

        # atur posisi window di tengah-tengah layar
        self.parent.geometry("%ix%i+%i+%i" % (lebar, tinggi,
                                              setengahLebar, setengahTinggi))

        # atur Image via Komponen Label
        Tk.Label(self.parent, image=self.imgSplash).pack()


if __name__ == '__main__':
    root = Tk.Tk()

    root.overrideredirect(True)
    s = ttk.Style()
    s.theme_use('clam')
    s.configure("red.Horizontal.TProgressbar", foreground='black', background='black')
    progressbar = Progressbar(root,style="red.Horizontal.TProgressbar",orient=HORIZONTAL,length=10000, mode='determinate')
    progressbar.pack(side="bottom")
    app = DemoSplashScreen(root)
    progressbar.start()

    # menutup window setelah 5 detik
    root.after(5500, root.destroy)


    root.mainloop()
class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("")
        master.geometry('1600x900')
        master.configure(bg='#239ab9')
        self.stepTwo = Tk.Frame(master, width=1500, height=850, bg='#239ab9')
        self.stepTwo.grid(row=2, columnspan=7)
        self.font=tf.Font(family='Times',size=25,weight=tf.BOLD)

        self.label = Tk.Label(master, text="Offline Signature Detection",bg='#0084ff',font=self.font)

        self.label.grid(row=0,column=3,padx=3,pady=10)
        self.inputwindow = Tk.LabelFrame(self.stepTwo, text='input image', width=500, height=650, bg='#52bad5', font=self.font)
        self.inputwindow.grid(row=2, column=0, sticky='ns', padx=6)
        self.inputwindow.pack_propagate(0)
        self.outputwindow = Tk.LabelFrame(self.stepTwo, text='output', width=500, height=650, bg='#52bad5', font=self.font)
        self.outputwindow.grid(row=2, column=1, padx=3, pady=3, sticky='nsew')
        self.operations = Tk.LabelFrame(self.stepTwo, text='operations', width=300, height=550, font=self.font, bg='#52bad5')
        self.operations.grid(row=2, column=2, sticky='ns', padx=3, pady=3)
        helv36 = tf.Font(family='Helvetica', size=15, weight=tf.BOLD)
        self.Input = Tk.Button(self.inputwindow, text='Select Image', width=20, bg='#487eb0', font=helv36,command=self.input)
        self.Input.pack(side=Tk.BOTTOM, pady=5)

        self.button = Tk.Button(self.operations, width=30, text='Segmentation', bg='#487eb0', font=helv36,command=self.segmented )
        self.button.pack(side=Tk.TOP, pady=10)

        self.button1 = Tk.Button(self.operations, width=30, text='HOG Features', bg='#487eb0', font=helv36,command=self.features )
        self.button1.pack(side=Tk.TOP, pady=10)
        self.button2 = Tk.Button(self.operations, width=30, text='Naive Bayes', font=helv36, bg='#487eb0',command=self.NiaveBayes)
        self.button2.pack(side=Tk.TOP, pady=10)
        self.button3 = Tk.Button(self.operations, width=30, text='Decision Tree', font=helv36, bg='#487eb0',command=self.DecisionTree)
        self.button3.pack(side=Tk.TOP, pady=10)
        self.button4 = Tk.Button(self.operations, width=30, text='Classifier Comparison', font=helv36, bg='#487eb0',command=self.Comparison)
        self.button4.pack(side=Tk.TOP, pady=10)
        self.button.configure(state=Tk.DISABLED)
        self.button1.configure(state=Tk.DISABLED)
        self.button2.configure(state=Tk.DISABLED)
        self.button3.configure(state=Tk.DISABLED)
        self.button4.configure(state=Tk.DISABLED)


    def input(self):
        global panelA
        path = fd.askopenfilename()
        self.img=cv2.imread(path)
        imff = Image.fromarray(self.img)
        imgt = ImageTk.PhotoImage(image=imff)
        self.button.configure(state=Tk.NORMAL)
        self.button1.configure(state=Tk.NORMAL)
        self.button2.configure(state=Tk.NORMAL)
        self.button3.configure(state=Tk.NORMAL)
        self.button4.configure(state=Tk.NORMAL)

        if panelA is None:
            panelA = Tk.Label(self.inputwindow, image=imgt)
            panelA.image = imgt
            panelA.pack(side="left", padx=10, pady=10)
        else:
            panelA.configure(image=imgt)
            panelA.image = imgt


    def segmented(self):
        for widget in self.outputwindow.winfo_children():
            widget.destroy()
        global panelB
        smg=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.ret, self.thresh = cv2.threshold(smg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        imff = Image.fromarray(self.thresh)
        imgt = ImageTk.PhotoImage(image=imff)
        l = Tk.Label(self.outputwindow, text="Segmented Image", bg="gray")
        l.place(x=25, y=25, anchor="center")
        l.pack(side=Tk.TOP, padx=10)
        panelB = Tk.Label(self.outputwindow, image=imgt)
        panelB.image = imgt
        panelB.pack(side=Tk.TOP, padx=14, pady=14)

    def features(self):
        for widget in self.outputwindow.winfo_children():
            widget.destroy()
        global panelC
        [H, hogImage] = feature.hog(self.thresh, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys',
                                    cells_per_block=(2, 2),
                                    transform_sqrt=True, visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        imff = Image.fromarray(hogImage)
        imgt = ImageTk.PhotoImage(image=imff)
        l = Tk.Label(self.outputwindow, text="Hog Features", bg="gray")
        l.place(x=25, y=25, anchor="center")
        l.pack(side=Tk.TOP, padx=10)
        panelC = Tk.Label(self.outputwindow, image=imgt)
        panelC.image = imgt
        panelC.pack(side=Tk.TOP, padx=14, pady=14)
    def NiaveBayes(self):
        global panelD
        for widget in self.outputwindow.winfo_children():
            widget.destroy()
        self.img=Image.open('naive bayes.PNG')
        self.img = self.img.resize((600, 600), Image.ANTIALIAS)

        imgt = ImageTk.PhotoImage(image=self.img)
        panelD = Tk.Label(self.outputwindow, image=imgt)
        panelD.image = imgt
        panelD.pack(side=Tk.TOP, padx=14, pady=14)
    def DecisionTree(self):
        global panelE
        for widget in self.outputwindow.winfo_children():
            widget.destroy()
        self.img = Image.open('DecisionTree.PNG')
        self.img = self.img.resize((600, 600), Image.ANTIALIAS)

        imgt = ImageTk.PhotoImage(image=self.img)
        panelE = Tk.Label(self.outputwindow, image=imgt)
        panelE.image = imgt
        panelE.pack(side=Tk.TOP, padx=14, pady=14)
    def Comparison(self):
        global panelF
        for widget in self.outputwindow.winfo_children():
            widget.destroy()
        self.img = Image.open('Capture.PNG')
        self.img = self.img.resize((600, 600), Image.ANTIALIAS)

        imgt = ImageTk.PhotoImage(image=self.img)
        panelF = Tk.Label(self.outputwindow, image=imgt)
        panelF.image = imgt
        panelF.pack(side=Tk.TOP, padx=14, pady=14)








panelA=None
panelB=None
panelC=None
panelD=None
panelE=None
panelF=None
root = Tk.Tk()
my_gui = MyFirstGUI(root)

root.mainloop()