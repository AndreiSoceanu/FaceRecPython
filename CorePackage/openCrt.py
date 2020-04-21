import os
import pickle
import sys

import cv2
import dlib
import numpy as np

cw = int(sys.argv[1])
fw = int(sys.argv[2])
fb = int(sys.argv[3])
fs = int(sys.argv[4])

print(cw, fw, fb, fs)

video_capture = cv2.VideoCapture(0)  # 1 or 2 for usb 0 for default
video_capture.set(cv2.CAP_PROP_FPS, 60)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('../ConfigData/shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('../ConfigData/dlib_face_recognition_resnet_model_v1.dat')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(cw, cw))

# TOLERANCE should be around 0.5 - 0.6
TOLERANCE = 0.6


def get_face_encdescriptor(image):
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


# This function takes a list of known faces
def compare_face_descriptors(known_faces, face):
    return np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE


def find_match(known_faces, names, face):
    matches = compare_face_descriptors(known_faces, face)
    count = 0
    for match in matches:
        if match:
            return names[count]
        count += 1
    return "unknown"


encodings_names = os.listdir('../Encodings')

known_encodings = []

for name in encodings_names:
    with open("../Encodings/" + name, 'rb') as f:
        v = pickle.load(f)
    known_encodings.extend(v)
# print(known_encodings)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)

    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    frame = cv2.bilateralFilter(frame, fw, fb, fs)

    size = frame.shape
    focal_length = size[1]
    center = (int(size[1] / 2), int(size[0] / 2))
    faces = face_detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        ROI = frame[y1:y2, x1:x2]
        encs = get_face_encdescriptor(ROI)
        if len(encs) == 0:
            continue
        encoding = encs[0]
        match = find_match(known_encodings, encodings_names, encoding)
        if match == "unknown":
            match = "unknown.data"  # fast workaround for next line
        cv2.putText(frame, match[:-5], (x1, y1), fontScale=0.5, color=(255, 255, 255), lineType=3,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break
video_capture.release()
cv2.destroyAllWindows()
