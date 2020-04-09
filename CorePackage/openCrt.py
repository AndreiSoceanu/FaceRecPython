import os
import pickle

import cv2
import dlib
import numpy as np

video_capture = cv2.VideoCapture(0)  # 1 or 2 for usb 0 for default
video_capture.set(cv2.CAP_PROP_FPS, 60)

# Get Face Detector from dlib
# This allows us to detect faces in images
face_detector = dlib.get_frontal_face_detector()
# Get Pose Predictor from dlib
# This allows us to detect landmark points in faces and understand the pose/angle of the face
shape_predictor = dlib.shape_predictor('../ConfigData/shape_predictor_68_face_landmarks.dat')
# Get the face recognition model
# This is what gives us the face encodings (numbers that identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('../ConfigData/dlib_face_recognition_resnet_model_v1.dat')
# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.6


# This function will take an image and return its face encodings using the neural network
def get_face_encodings(image):
    # Detect faces using the face detector
    detected_faces = face_detector(image, 1)

    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people,
    # regardless of camera angle and/or face positioning in the image
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # Return an array with True/Face values based on whether or not a known face matched with the given face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the
    # TOLERANCE value
    return np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE


# This function returns the name of the person whose image matches with the given face (or 'Not Found')
# known_faces is a list of face encodings
# names is a list of the names of people (in the same order as the face encodings - to match the name with an encoding)
# face is the face we are looking for
def find_match(known_faces, names, face):
    # Call compare_face_encodings to get a list of True/False values indicating whether or not there's a match
    matches = compare_face_encodings(known_faces, face)
    # Return the name of the first match
    # print(matches)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    # Return not found if no match found
    return "unknown"


# Get path to all the known images
# Filtering on .jpg extension - so this will only work with JPEG images ending with .jpg
encodings_names = os.listdir('../Encodings')

known_encodings = []

for name in encodings_names:
    with open("../Encodings/" + name, 'rb') as f:
        v = pickle.load(f)
    known_encodings.extend(v)
print(known_encodings)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    size = frame.shape
    focal_length = size[1]
    center = (int(size[1] / 2), int(size[0] / 2))
    faces = face_detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 7)
        ROI = frame[y1:y2, x1:x2]
        encs = get_face_encodings(ROI)
        if len(encs) == 0:
            continue
        encoding = encs[0]
        match = find_match(known_encodings, encodings_names, encoding)
        if match == "unknown":
            match = "unknown.data"
        cv2.putText(frame, match[:-5], (x1, y1), fontScale=0.5, color=(255, 255, 255), lineType=3,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX)
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
