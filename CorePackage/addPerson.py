import math
import time

import cv2
import dlib
import numpy as np

video_capture = cv2.VideoCapture(0)  # 1 or 2 for usb 0 for default
video_capture.set(cv2.CAP_PROP_FPS, 60)

[faceCenter, faceLeft, faceRight, faceUp, faceDown] = [False, False, False, False, False]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../ConfigData/shape_predictor_68_face_landmarks.dat")
counter = 0
t = 0
done = False
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    if not faceLeft:
        leftRct = cv2.rectangle(frame, (0, 251), (120, 233), (0, 0, 0), -1)
        cv2.putText(frame, "LOOK LEFT", (0, 250), fontScale=0.7, color=(255, 255, 255), lineType=3,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX)

    if not faceRight:
        rightRct = cv2.rectangle(frame, (500, 251), (630, 233), (0, 0, 0), -1)
        cv2.putText(frame, "LOOK RIGHT", (500, 250), fontScale=0.7, color=(255, 255, 255), lineType=3,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX)

    if not faceUp:
        upRct = cv2.rectangle(frame, (250, 0), (350, 20), (0, 0, 0), -1)
        cv2.putText(frame, "LOOK UP", (250, 15), fontScale=0.7, color=(255, 255, 255), lineType=3,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX)

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
        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        alpha = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        alpha = math.degrees(alpha)
        # print(alpha)

        # FACE RIGHT
        if (p2[0] - p1[0] > 0 and dist > 320 and -15 < alpha < 15):
            # print("RIGHT")
            faceRight = True
            # TODO SAVE IMAGE

        # ---
        # FACE LEFT
        if (p1[0] - p2[0] > 0 and dist > 320 and -195 < alpha < -165):
            # print("LEFT")
            faceLeft = True
            # TODO SAVE IMAGE
        # ---

        # FACE UP
        if (p1[1] - p2[1] > 0 and dist > 180 and -75 > alpha > -105):
            # print("UP")
            faceUp = True
            # TODO SAVE IMAGE
        # ---

        # TURN DOWN FOR WATT ????
        # if (p2[0] - p1[0] < 0 and dist > 150 and 85 < alpha < 105):
        #     #print("DOWN")
        #     faceDown = True
        #     # TODO SAVE IMAGE
        # # ---

        ### FILL BASE
        if (faceUp and faceLeft and faceRight):
            millis = int(round(time.time() * 1000))
            if (millis - t > 1000):
                t = millis
                print("BAM")
                counter = counter + 1
                if (counter == 4):
                    # done = True
                    break
        if (done):
            break
    if (done):
        break
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 27:
        break
# When everything is done, release the capture and destroy the windows, so memory will be free
video_capture.release()
cv2.destroyAllWindows()
