
import tkinter as Tk
import tkinter.filedialog as fd
import tkinter.font as tf
import cv2
import numpy as np


from PIL import Image
from PIL import ImageTk
from skimage import exposure
from skimage import feature

root.title('')
root.configure(bg='#192a56')
root.geometry("1670x850")
root.resizable(0, 0)
stepTwo = Tk.Frame(root, width=1580, height=850, bg='#192a56')
stepTwo.grid(row=2, columnspan=7)
helv35 = tf.Font(family='Times', size=20, weight=tf.BOLD)
l = Tk.Label(stepTwo, text="Offline Signature Verification", bg="#2d98da",font=helv35)
l.grid(row=0,column=1,pady=23)
inputwindow= Tk.LabelFrame(stepTwo, text='input image', width=500, height=650, bg='#273c75',font=helv35)
inputwindow.grid(row=2,column=0,sticky='ns',padx=6)
inputwindow.pack_propagate(0)
outputwindow= Tk.LabelFrame(stepTwo, text='output', width=500, height=650, bg='#273c75',font=helv35)
outputwindow.grid(row=2,column=1,padx=3,pady=3,sticky='nsew')
operations= Tk.LabelFrame(stepTwo, text='operations', width=300, height=550,font=helv35,bg='#273c75')
operations.grid(row=2,column=2,sticky='ns',padx=3,pady=3)
operations.pack_propagate(0)
global handle
handle=[]

def select_image():


 global panelA, panelB
 path = fd.askopenfilename()
 self.handle.append(path)
 img=cv2.imread(handle[0])
 input.append(img)
 imff = Image.fromarray(img)
 imgt = ImageTk.PhotoImage(image=imff)





 if panelA is None:
   panelA = Tk.Label(inputwindow,image=imgt)
   panelA.image = imgt
   panelA.pack(side="left", padx=10, pady=10)
   button.configure(state=Tk.NORMAL)
   button1.configure(state=Tk.NORMAL)
   button2.configure(state=Tk.NORMAL)
   button3.configure(state=Tk.NORMAL)
   button4.configure(state=Tk.NORMAL)
 else:
   panelA.configure(image=imgt)
   panelA.image = imgt


def segmented():
 for widget in outputwindow.winfo_children():
  widget.destroy()
  img = cv2.imread(handle[0])
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  imff = Image.fromarray(thresh)
  imgt = ImageTk.PhotoImage(image=imff)
  l = Tk.Label(outputwindow, text="Segmented Image", bg="gray")
  l.place(x=25, y=25, anchor="center")
  l.pack(side=Tk.TOP,padx=10)
  label2=Tk.Label(outputwindow, image=imgt)
  label2.image=imgt
  label2.pack(side=Tk.TOP, padx=14, pady=14)
def features():
 for widget in outputwindow.winfo_children():
  widget.destroy()


 [H, hogImage] = feature.hog(thresh, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys', cells_per_block=(2, 2),
                             transform_sqrt=True, visualise=True)
 hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
 hogImage = hogImage.astype("uint8")
 imff = Image.fromarray(hogImage)
 imgt = ImageTk.PhotoImage(image=imff)
 l = Tk.Label(outputwindow, text="Hog Features", bg="gray")
 l.place(x=25, y=25, anchor="center")
 l.pack(side=Tk.TOP, padx=10)
 label2 = Tk.Label(outputwindow, image=imgt)
 label2.image = imgt
 label2.pack(side=Tk.TOP, padx=14, pady=14)





def Classification():
 for widget in outputwindow.winfo_children():
  widget.destroy()

 global panelB
 im = cv2.imread('w3.png')
 image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
 cls = Image.fromarray(image)
 cls = ImageTk.PhotoImage(image=cls)
 panelB=Tk.Label(outputwindow, image=cls)
 panelB.image=cls
 panelB.pack(side=Tk.TOP, padx=12, pady=12)
def Comparison():
 for widget in outputwindow.winfo_children():
  widget.destroy()

 global panelC
 im = cv2.imread('Capture.png')
 image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
 cls = Image.fromarray(image)
 cls = ImageTk.PhotoImage(image=cls)
 panelC=Tk.Label(outputwindow, image=cls)
 panelC.image=cls
 panelC.pack(side=Tk.TOP, padx=12, pady=12)
def Classrep():
 for widget in outputwindow.winfo_children():
  widget.destroy()

 global panelD
 im = cv2.imread('w.png')
 image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
 cls = Image.fromarray(image)
 cls = ImageTk.PhotoImage(image=cls)
 panelD=Tk.Label(outputwindow, image=cls)
 panelD.image=cls
 panelD.pack(side=Tk.TOP, padx=12, pady=12)


panelA=None
panelB=None
panelC=None
panelD=None
handle=None

helv36 = tf.Font(family='Helvetica', size=15, weight=tf.BOLD)
button = Tk.Button(operations, width=50, text='Segmentation', bg='#487eb0', font=helv36, command=segmented)
button.pack(side=Tk.TOP, pady=10)
button1 = Tk.Button(operations, width=50, text='HOG Features', bg='#487eb0', font=helv36, command=features)
button1.pack(side=Tk.TOP, pady=10)
button2 = Tk.Button(operations, width=50, text='Classification', font=helv36, bg='#487eb0', command=Classification)
button2.pack(side=Tk.TOP, pady=10)
button3 = Tk.Button(operations, width=50, text='Comparison Graph', font=helv36, bg='#487eb0', command=Comparison)
button3.pack(side=Tk.TOP, pady=10)
button4 = Tk.Button(operations, width=50, text='Classification Report', font=helv36, bg='#487eb0', command=Classrep)
button4.pack(side=Tk.TOP, pady=10)
Input = Tk.Button(inputwindow, text='Select Image', width=20, bg='#487eb0',font=helv35, command=select_image )
Input.pack(side=Tk.BOTTOM, pady=5)

root=Tk.Tk()
root.mainloop()


