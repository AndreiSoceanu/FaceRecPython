import os
import pickle

import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../ConfigData/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("../ConfigData/dlib_face_recognition_resnet_model_v1.dat")

origin = open("../ConfigData/persons.data", "r")
lines = origin.readlines()
n = int(lines[0])
lines.pop(0)
fileNames = []
keys = []
encodings = []


# This function will take an image and return its face encodings using the neural network
def get_face_encodings(img, face):
    # Detect faces using the face detector
    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people,
    # regardless of camera angle and/or face positioning in the image
    shapes_faces = predictor(img, face)
    # For every face detected, compute the face encodings
    return np.array(face_recognition_model.compute_face_descriptor(img, shapes_faces, 1))


for line in lines:
    name = line.replace('\n', '')

    files = os.listdir("../TrainBase/" + name)
    fileNames.extend(files)
    size = len(files)
    for i in range(0, size):

        img = cv2.imread("../TrainBase/" + name + "/" + files[i])
        face = detector(img, 1)
        if len(face) == 0:
            continue
        else:
            encodings.append(get_face_encodings(img, face[0]))
            keys.append(name)

with open("../ConfigData/trainResult.data", "wb") as f:
    pickle.dump(encodings, f)

with open("../ConfigData/names.data", "wb") as f2:
    pickle.dump(keys, f2)

print(keys)
print(len(fileNames))
