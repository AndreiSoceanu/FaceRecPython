import copy
import pickle
import sys

import cv2
import dlib
import numpy as np

name = str(sys.argv[1])
path = "../TrainBase/"

video_capture = cv2.VideoCapture(0)  # 1 or 2 for usb 0 for default
video_capture.set(cv2.CAP_PROP_FPS, 60)
face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1('../ConfigData/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('../ConfigData/shape_predictor_68_face_landmarks.dat')


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


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    copyFrame = copy.deepcopy(frame)

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
    faces = face_detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 7)
        ROI = frame[y1:y2, x1:x2]
        landmarks = shape_predictor(frame, face)
        noseTip = (landmarks.part(30).x, landmarks.part(30).y)
        chin = (landmarks.part(8).x, landmarks.part(8).y)
        leftEyeLeftCorner = (landmarks.part(36).x, landmarks.part(36).y)
        rightEyeRightCorner = (landmarks.part(45).x, landmarks.part(45).y)
        leftMouthCorner = (landmarks.part(48).x, landmarks.part(48).y)
        rightMouthCorner = (landmarks.part(54).x, landmarks.part(54).y)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

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

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        break
    elif k % 256 == 32:
        # SPACE pressed
        if len(faces) != 1:
            # msg
            continue
        else:
            encoding = get_face_encodings(copyFrame)
            with open("../Encodings/" + name + ".data", 'wb') as f:
                pickle.dump(encoding, f)

            img_name = "../TrainBase/" + name + ".jpg"
            cv2.imwrite(img_name, frame)
            break
# When everything is done, release the capture and destroy the windows, so memory will be free
video_capture.release()
cv2.destroyAllWindows()
