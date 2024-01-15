import os
import cv2
import dlib
import time
import numpy as np
detector = dlib.get_frontal_face_detector()
# load the facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_81_face_landmarks.dat')

# initialize the video capture
cap = cv2.VideoCapture(0)

# define the eye aspect ratio function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# set the blink threshold and consecutive frames counter
BLINK_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 3
blink_counter = 0
consecutive_frames = 0
BLINK = False

# get the desktop folder path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# start the main loop
while True:
    # read a frame from the video capture
    ret, frame = cap.read()

    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # detect the faces using dlib
    faces = detector(gray, 0)

    # loop over each detected face
    for face in faces:
        # get the facial landmarks
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # compute the eye aspect ratio for the left and right eyes
        left_eye = landmarks[42:48]
        right_eye = landmarks[36:42]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # compute the average eye aspect ratio
        ear = (left_ear + right_ear) / 2.0

        # check if the eye aspect ratio is below the blink threshold
        if ear < BLINK_THRESHOLD:
            consecutive_frames += 1
        else:
            if consecutive_frames >= CONSECUTIVE_FRAMES:
                blink_counter += 1
                BLINK = True
            consecutive_frames = 0

        # draw the landmarks and the blink counter on the frame
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(frame, "Blinks: {}".format(blink_counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the current frame
    cv2.imshow("Frame", frame)

    # if the user has blinked, capture an image
    if BLINK:
        # get the current timestamp
        timestamp = int(time.time())

        # construct the file name for the image and save it
        file_name = "blink_{}.png".format(timestamp)
        file_path = os.path.join(desktop_path, file_name)
        cv2.imwrite(file_path, frame)

    # reset the blink flag
    BLINK = False

    # check for key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()