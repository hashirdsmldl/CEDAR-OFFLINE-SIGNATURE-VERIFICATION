



handle=[]
path=fd.askopenfilename(title='Open File')


handle.append(path)

path2=cv2.imread(handle[0])
img=cv2.cvtColor(path2,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('imae',thresh)



[H, hogImage] = feature.hog(thresh, orientations=9, pixels_per_cell=(8, 8), block_norm='L2-Hys', cells_per_block=(2, 2),
                             transform_sqrt=True, visualise=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow('hogImage',hogImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
images = [cv2.imread(file) for file in glob.iglob("D:\Project Folder\CEDAR\Training\*.png",recursive=True)]
print(len(images))
cv2.imshow("image",images[0])

#Label
ClassLabel=['forgered','original']
ClassLabel12=24*ClassLabel
print(len(ClassLabel12))

ClassLabel12.sort()


#print(len(result))


#segmentation
segmented=[]
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    print(type(thresh))
    segmented.append(thresh)

cv2.imshow("segmented image",segmented[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
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
cv2.imshow("feature extracted",features[2])

mean=[]
for img in features:

    x=np.mean(img)
    mean.append(x)



#Mean of  image at 10 location


df=pd.DataFrame(list(zip(ImageId,mean,ClassLabel12)), columns=['ID','Features','Class'])
print(df)
features1= list(df.columns[:2])
print(features1)


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
print(pred)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print(accuracy_score(test_labels, pred))
# Initialize our classifier
gnb = GaussianNB()
# Train our classifier
model = gnb.fit(train, train_labels)


# Make predictions
preds = gnb.predict(test)
print(preds)

cm = confusion_matrix(test_labels,preds)
TP =cm[0][0]
TN= cm[0][1]
FP=cm[1][0]
FN=cm[1][1]
print(TP)
print(TN)
print(FP)
print(FN)
print(accuracy_score(test_labels, preds))

