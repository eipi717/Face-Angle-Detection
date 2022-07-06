# Detect faces and crop it to new file [face{i}] for i = 1, 2, 3, ...

import cv2

def Detection(img):
    # Read the input image
    img = cv2.imread(img)
    #img = imutils.resize(img, width = 900)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('/Users/nicholas717/opt/anaconda3/pkgs/opencv-4.5.5-py38hc2a0b3f_0/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    i = 1

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        face_mid = ((x + x + w + 5) // 2, (y + y + h + 2) // 2)

        # Crop the face from the original image
        faces = img[y-20:y + h + 10, x:x + w + 10]
        cv2.imwrite('./test_image/face' + str(i) + '.jpg', faces)
        i += 1

    # Display the output
    cv2.imwrite('./test_image/detcted.png', img)
    return
