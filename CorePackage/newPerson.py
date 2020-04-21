import copy
import pickle
import sys

import cv2
import dlib
import numpy as np

name = str(sys.argv[1])
cw = int(sys.argv[2])
fw = int(sys.argv[3])
fb = int(sys.argv[4])
fs = int(sys.argv[5])

# print("Test")

path = "../TrainBase/"

video_capture = cv2.VideoCapture(0)  # 1 or 2 for usb 0 for default
video_capture.set(cv2.CAP_PROP_FPS, 60)
face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1('../ConfigData/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('../ConfigData/shape_predictor_68_face_landmarks.dat')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(cw, cw))


def get_face_encodings(image):
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    copyFrame = copy.deepcopy(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    lab_planes = cv2.split(lab)

    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    frame = cv2.bilateralFilter(frame, fw, fb, fs)

    reference_pts = np.array([
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
    camera_intrinsic_params = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))
    faces = face_detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        ROI = frame[y1:y2, x1:x2]
        markers = shape_predictor(frame, face)
        noseTip = (markers.part(30).x, markers.part(30).y)
        chin = (markers.part(8).x, markers.part(8).y)
        leftEyeLeftCorner = (markers.part(36).x, markers.part(36).y)
        rightEyeRightCorner = (markers.part(45).x, markers.part(45).y)
        leftMouthCorner = (markers.part(48).x, markers.part(48).y)
        rightMouthCorner = (markers.part(54).x, markers.part(54).y)

        for n in range(0, 68):
            x = markers.part(n).x
            y = markers.part(n).y
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        image_points = np.array([
            noseTip,  # Nose tip
            chin,  # Chin
            leftEyeLeftCorner,  # Left eye left corner
            rightEyeRightCorner,  # Right eye right corner
            leftMouthCorner,  # Left Mouth corner
            rightMouthCorner  # Right mouth corner
        ], dtype="double")

        (success, rotation_vector, translation_vector) = cv2.solvePnP(reference_pts, image_points,
                                                                      camera_intrinsic_params,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector, camera_intrinsic_params, dist_coeffs)

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

            continue
        else:
            encoding = get_face_encodings(copyFrame)

            with open("../Encodings/" + name + ".data", 'wb') as f:
                pickle.dump(encoding, f, protocol=2)
                print("Done")

            break
# When everything is done, release the capture and destroy the windows, so memory will be free
video_capture.release()
cv2.destroyAllWindows()
print(".")
