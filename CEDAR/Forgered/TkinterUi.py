import cv2
import matplotlib
import tkinter as Tk
import tkinter.filedialog as fd
import tkinter.font as tf
import cv2
import numpy as np


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
images = [cv2.imread(file) for file in glob.iglob("D:\Project Folder\CEDAR\Training\*.png", recursive=True)]
ClassLabel = ['forgered', 'original']
ClassLabel12 = 24 * ClassLabel
ClassLabel12.sort()

segmented = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    segmented.append(thresh)

ImageId = []
for i in range(len(images)):
     ImageId.append(i)

features = []
for img in segmented:
    [H, hogImage] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys',
                                        cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    features.append(hogImage)
    mean = []
for img in features:
    x = np.mean(img)
    mean.append(x)
    df = pd.DataFrame(list(zip(ImageId, mean, ClassLabel12)), columns=['ID', 'Features', 'Class'])
    features1 = list(df.columns[:2])
    label = df['Class']
    feat = df[features1]
    from sklearn.model_selection import train_test_split
    train, test, train_labels, test_labels = train_test_split(feat, label,test_size=0.3 ,random_state=42)
    dt = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
    dt.fit(train, train_labels)
    pred = dt.predict(test)
    a = accuracy_score(test_labels, pred)
    b = confusion_matrix(test_labels, pred)
    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    model = gnb.fit(train, train_labels)
    preds = gnb.predict(test)
    c = confusion_matrix(test_labels, preds)
    d = accuracy_score(test_labels, preds)
    cm_df = pd.DataFrame(b,
                                          index=['Forgered', 'Original'],
                                          columns=['Forgered', 'Original'])
    cn_df = pd.DataFrame(c,
                                          index=['Forgered', 'Original'],
                                          columns=['Forgered', 'Original'])
    from sklearn.metrics import accuracy_score, log_loss
    log_cols = ["Classifier", "Accuracy", "LogLoss"]
    log = pd.DataFrame(columns=log_cols)

    classifiers = [GaussianNB(), tree.DecisionTreeClassifier(), RandomForestClassifier()]
    for clf in classifiers:
        clf.fit(train, train_labels)
        name = clf.__class__.__name__
        preds = clf.predict(test)
        acc = accuracy_score(test_labels, preds)
        pred = clf.predict_proba(test)
        ll = log_loss(test_labels, pred)
        print("Log Loss: {}".format(ll))
        log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
        log = log.append(log_entry)
        print("=" * 30)


def Segmented():
    cv2.imshow("segmented image", segmented[1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Features():

    cv2.imshow("feature extracted", features[2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Classifier1():
 plt.figure(figsize=(5.5, 4))
 sns.heatmap(cm_df, annot=True)
 plt.title('Tree \n Accuracy:{0:.3f}'.format(a))
 plt.ylabel('True label')
 plt.xlabel('Predicted label')
 plt.show()
def Classifier2():
 plt.figure(figsize=(5.5, 4))
 sns.heatmap(cn_df, annot=True)
 plt.title('Naive Base \n Accuracy:{0:.3f}'.format(d))
 plt.ylabel('True label')
 plt.xlabel('Predict'
            'ed label')
 plt.show()
 def Comparison():
    sns.set_color_codes("muted")
    sns.barplot(x='Classifier', y='Accuracy', data=log, color="b")

    plt.ylabel('Accuracy %')
    plt.title('Classifier Accuracy')
    plt.show()

global comparison
def Comparison1():
    sns.set_color_codes("muted")
    sns.barplot(x='LogLoss', y='Classifier', data=log, color="b")

    plt.xlabel('LogLoss %')
    plt.title('Classifier Accuracy')

    plt.savefig('abc.png')
    plt.show()
def classification():
    viz = ClassificationReport(GaussianNB())
    viz.fit(train, train_labels)
    viz.score(test, test_labels)
    viz.poof()


def Input_Image():


    cv2.imshow("image", images[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()



root = tk.Tk()
panelA=None
panelB=None



root.configure(background='lightgray')
root.title("OSV tool")
root.geometry("800x200")



group = tk.LabelFrame(root, text="Input Commands",background = 'orange',width=750, height=200, padx=5, pady=5)

group.pack()

button1 = tk.Button(group,
                   text="Input Image",width=16,command=Input_Image
                   )
button1.grid(row=0,column=0)
button2 = tk.Button(group,
                   text="Segmented",width=16,command=Segmented
                   )
button2.grid(row=1,column=0)
button3 = tk.Button(group,
                   text="HOG Features",width=16,command=Features
                  )
button3.grid(row=2,column=0)
button4 = tk.Button(group,
                   text="Naive Bayes Classsifier",width=16,command=Classifier2

                   )
button4.grid(row=3,column=0)
button5 = tk.Button(group,
                   text="Decision Tree Classifier",width=16,command=Classifier1
                   )
button5.grid(row=4,column=0)
button6 = tk.Button(group,
                   text="Accuracy Comparison",width=15,command=Comparison
                   )
button6.grid(row=5,column=0)
button6 = tk.Button(group,
                   text="Log Loss comparison",width=15,command=Comparison1
                   )
button6.grid(row=6,column=0)
button7 = tk.Button(group,
                   text="Classification Report",width=15,command=classification
                   )
button6.grid(row=6,column=0)

root.mainloop()