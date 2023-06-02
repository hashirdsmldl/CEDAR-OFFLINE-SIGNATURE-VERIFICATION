
import cv2
import numpy as np

import glob

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

images = [cv2.imread(file) for file in glob.iglob("D:\Project Folder\CEDAR\Training\*.png",recursive=True)]



#Label
ClassLabel=['forgered','original']
ClassLabel12=24*ClassLabel


ClassLabel12.sort()


#print(len(result))


#segmentation
segmented=[]
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    segmented.append(thresh)



ImageId=[]
for i in range(len(images)):
    ImageId.append(i)




#Feature Extraction

from skimage import exposure
from skimage import feature

import pandas as pd
features=[]

for img in segmented:
    [H, hogImage] = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys', cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    features.append(hogImage)


mean=[]
for img in features:

    x=np.mean(img)
    mean.append(x)



#Mean of  image at 10 location


df=pd.DataFrame(list(zip(ImageId,mean,ClassLabel12)), columns=['ID','Features','Class'])

features1= list(df.columns[:2])
print(df)


label= df['Class']
feat = df[features1]

train, test, train_labels, test_labels = train_test_split(feat,
                                                          label,
                                                          test_size=0.3
                                                          ,
                                                          random_state=42)
dt = tree.DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(train, train_labels)
pred=dt.predict(test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Initialize our classifier
gnb = GaussianNB()
# Train our classifier
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)

# Make predictions






cm = confusion_matrix(test_labels,preds)

TP =cm[0][0]
TN= cm[0][1]
FP=cm[1][0]
FN=cm[1][1]
print(pred)
a=accuracy_score(test_labels, pred)
print(accuracy_score(test_labels, pred))
print(preds)
b=accuracy_score(test_labels, preds)
print(accuracy_score(test_labels, preds))
print(TP)
print(TN)
print(FP)
print(FN)

plt.figure(figsize=(5, 5))
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Forgered','Original']
plt.title('Naive Base \n Accuracy:{0:.3f}'.format(b))
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
b = confusion_matrix(test_labels,pred)

cm_df = pd.DataFrame(b,
                                          index=['Forgered', 'Original'],
                                          columns=['Forgered', 'Original'])

plt.figure(figsize=(5.5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Tree \n Accuracy:{0:.3f}'.format(a))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()