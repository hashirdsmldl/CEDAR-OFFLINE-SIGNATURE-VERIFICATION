from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import cv2
def segmented(self):
    global panelB
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = Image.fromarray(thresh)

    # ...and then to ImageTk format
    thresh = ImageTk.PhotoImage(thresh)
    if panelB is None:
        # the first panel will store our original image
        panelB = Label(image=thresh)
        panelB.thresh = thresh
        panelB.pack(side="left", padx=10, pady=10)
    else:
        panelB.configure(image=self.thresh)
        panelB.thresh = thresh


def select_image(self):
    # grab a reference to the image panels
    global panelA


    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkFileDialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)


        #  represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)


        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)


        # if the panels are None, initialize them
        if panelA is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map


        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=self.image)
            panelA.image = image



# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = Button(root, text="Select an image", command=lambda :select_image(self=NORMAL))
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn = Button(root, text="Segmented", command=lambda :segmented(self=NORMAL))
btn.pack(side="bottom", fill="both", expand="yes", padx="12", pady="12")

# kick off the GUI
root.mainloop()