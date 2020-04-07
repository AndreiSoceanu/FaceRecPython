import os

import cv2
import dlib
import numpy as np

video_capture = cv2.VideoCapture(0)  # 1 or 2 for usb 0 for default
video_capture.set(cv2.CAP_PROP_FPS, 60)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../ConfigData/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("../ConfigData/dlib_face_recognition_resnet_model_v1.dat")

# This is the tolerance for face comparisons
# The lower the number - the stricter the comparison
# To avoid false matches, use lower value
# To avoid false negatives (i.e. faces of the same person doesn't match), use higher value
# 0.5-0.6 works well
TOLERANCE = 0.6

origin = open("../ConfigData/persons.data", "r")
lines = origin.readlines()
n = int(lines[0])
lines.pop(0)
fileNames = []
keys = []
encodings = []


# This function will take an image and return its face encodings using the neural network
def get_face_encodings(img):
    # Detect faces using the face detector
    detected_faces = detector(img, 1)

    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people,
    # regardless of camera angle and/or face positioning in the image
    shapes_faces = [predictor(img, face) for face in detected_faces]
    # For every face detected, compute the face encodings
    return [np.array(face_recognition_model.compute_face_descriptor(img, face_pose, 1)) for face_pose in shapes_faces]


# This function takes a list of known faces
def compare_face_encodings(known_faces, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    # Calculate norm for the differences with each known face
    # Return an array with True/Face values based on whether or not a known face matched with the given face
    # A match occurs when the (norm) difference between a known face and the given face is less than or equal to the
    # TOLERANCE value
    return np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE


for line in lines:
    name = line.replace('\n', '')

    files = os.listdir("../TrainBase/" + name)
    fileNames.extend(files)
    size = len(files)
    for i in range(0, size):
        keys.append(name)
        img = cv2.imread("../TrainBase/" + name + "/" + files[i])
        encodings.append(get_face_encodings(img))
# TODO filewrite encodings
print((encodings))
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, d=15, sigmaColor=15, sigmaSpace=75)
    # cv2.imshow("Gray", gray)
    # 3D model points.
    # ALEA BUNE
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    size = frame.shape
    focal_length = size[1]
    center = (int(size[1] / 2), int(size[0] / 2))
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))
    P1 = (center[0] - 200, center[1] - 175)
    P2 = (center[0] + 200, center[1] + 175)
    # cv2.circle(frame,P1, 10, (0, 255,0 ), -1)
    # cv2.circle(frame, P2, 10, (0, 255,0 ), -1)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 7)
        ROI = frame[y1:y2, x1:x2]
        landmarks = predictor(gray, face)
        noseTip = (landmarks.part(30).x, landmarks.part(30).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)
        leftEyeLeftCorner = (landmarks.part(36).x, landmarks.part(36).y)
        rightEyeRightCorner = (landmarks.part(45).x, landmarks.part(45).y)
        leftMouthCorner = (landmarks.part(48).x, landmarks.part(48).y)
        rightMouthCorner = (landmarks.part(54).x, landmarks.part(54).y)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            # THS IS LANDMARK n(x,y) #cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
        # Head Pose Est
        image_points = np.array([
            noseTip,  # Nose tip
            chin,  # Chin
            leftEyeLeftCorner,  # Left eye left corner
            rightEyeRightCorner,  # Right eye right corner
            leftMouthCorner,  # Left Mouth corner
            rightMouthCorner  # Right mouth corner
        ], dtype="double")

        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_matrix, dist_coeffs)

    # END for face in faces

    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break
# When everything is done, release the capture and destroy the windows, so memory will be free
video_capture.release()
cv2.destroyAllWindows()
print("Done")
