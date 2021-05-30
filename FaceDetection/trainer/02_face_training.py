import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = './dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    "./Cascades/haarcascade_frontalface_default.xml")
# function to get the images and label data


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def getImagesAndLabels(path):
    dirlist = []
    for x in os.listdir(path):
        dirlist.append(x)
    imagePaths = []
    for y in dirlist:
        imagePaths.append([os.path.join(path+'/'+y, f)
                          for f in os.listdir(path+'/'+y)])
    img = cv2.imread(imagePaths[1][0], 0)
    imagePaths = flatten_list(imagePaths)
    print(img)
    print(imagePaths[0])
    faceSamples = []
    ids = []
    count = 0
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale

        img_numpy = np.array(PIL_img, 'uint8')
        id = (os.path.split(imagePath)[-1].split(".")[0])
        # print(id)
        count = count+1
        faces = detector.detectMultiScale(img_numpy)

        if(len(faces) == 0):
            continue
        print(count, imagePath, len(faces), id)
        for (x, y, w, h) in faces:
            ids.append(int(id))
            faceSamples.append(img_numpy[y:y+h, x:x+w])

            # print(count, faceSamples, id)
    # print(len(faceSamples), len(ids))
    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
# print(ids)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
# recognizer.save()
recognizer.write('trainer/trainer.yml')

# # Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(
    len(np.unique(ids))))
