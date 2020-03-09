import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR ="D:\DeepLear\CATSVSDOG\PetImages"
CATEGORIES = ["cat","dog"]

for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #my path to data images
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        IMG_SIZE = 80
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        #plt.imshow(img_array, cmap="gray")
        #plt.show()
        break
    break
#print(img_array) #showing original data for sample
#Shape training to reduce the size of data images

#plt.imshow(new_array, cmap="gray")
#plt.show()

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # my path to data images
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:

                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

            #plt.imshow(img_array, cmap="gray")
            #plt.show()
            #break
        #break
create_training_data()
print(len(training_data))
#Doing Shuffle so that I can provide diffent data at diff time
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

for features, lable in training_data:
    X.append(features)
    y.append(lable)

X = np.array(X).reshape(-1, 80, 80,1)
#np.save('features.npy',X)  #saving
#X=np.load('features.npy') #loading

import pickle
pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#to load data
pickle_in =open("X.pickle","rb")
X = pickle.load(pickle_in)
X[1]