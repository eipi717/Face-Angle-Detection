import cv2
import dlib
import numpy as np
import selfmath
import imutils
import os
from detectface import Detection

def DetectChin(img, p):

    # Load the detector
    detector = dlib.get_frontal_face_detector()

    # Load the predictor
    predictor = dlib.shape_predictor("/Users/nicholas717/PycharmProjects/FaceAngleDetection/shape_predictor_68_face_landmarks_GTX.dat")

    # read the image
    img = cv2.imread(str(img))
    img = imutils.resize(img, width=350, height=350)

    # Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    faces = detector(gray)
    for face in faces:
        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Loop through all the points
        X = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            X.append((x, y))

            # Find the mid-point of two eyebrows
            if n is 22:
                mid_pt = selfmath.MidPt(X[21], X[22])
                cv2.circle(img=img, center=(int(mid_pt[0]), int(mid_pt[1])),
                           radius=3, color=(0, 255, 255), thickness=1)
                line = np.array([np.array(mid_pt), np.array(X[8])], dtype=int)

            # Obtain the angle of right face and left face to the nose
            elif n is 33:
                x14 = np.array(X[14])
                x15 = np.array(X[15])
                x33 = np.array(X[33])
                x01 = np.array(X[1])
                x02 = np.array(X[2])

                m1433 = selfmath.GetSlope(x14, x33)
                m1533 = selfmath.GetSlope(x15, x33)
                m0133 = selfmath.GetSlope(x01, x33)
                m0233 = selfmath.GetSlope(x02, x33)

                # Find the face width
                x00 = np.array(X[0])
                x16 = np.array(X[16])

            # Draw a circle
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # ----- Face angle -----
        # Find the angle of the chin
        # By the slope of two lines on left and right
        angle_left = selfmath.LinesInAngle(m1433, m1533)
        angle_right = selfmath.LinesInAngle(m0133, m0233)

        m1 = selfmath.GetSlope(X[6], X[8])
        m2 = selfmath.GetSlope(X[8], X[10])
        face_width = selfmath.dist(x00, x16)
        angle = np.abs(selfmath.LinesInAngle(m1, m2))

        print(f"----- Face angle detection -----\n"
              f"The face angle is {angle}\n"
              f"The face width is {face_width}")

        # ----- Distortion checking -----
        print(f"----- Distortion checking -----\n"
              f"The left angle is {angle_left} and the right is {angle_right}")

        if (np.abs(angle_left - angle_right) > 5.0):
            if(angle_left > angle_right):
                print(f"The face turned left with angle {angle_left}")
            else:
                print(f"The face turned right with angle {angle_right}")
        else:
            print("No distortion detected!")

    if p == True:
        # show the image
        cv2.imshow("test", img)

        # Delay between every fram
        cv2.waitKey(delay=0)

    # Close all windows
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":

    # Crop each faces from the "big" image
    origin_img = "/Users/nicholas717/Downloads/Work/code/12321.webp"
    Detection(origin_img)

    # Test the cropped faces in test_image/ dir
    for filename in os.listdir('test_image'):
        # Ignore example.png and .DS_Store
        if filename not in ('detected.png', '.DS_Store'):
                DetectChin("/Users/nicholas717/PycharmProjects/FaceAngleDetection/test_image/"
                           + str(filename), True)

