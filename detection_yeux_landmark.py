import cv2
import dlib
import numpy as np
from imutils import face_utils


def run():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to connect to camera.")

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    while cap.isOpened():
        ret, frame = cap.read()
  
        if ret:
            face_rects = detector(frame, 0)

            if len(face_rects) > 0:
                shape = predictor(frame, face_rects[0])
                shape = face_utils.shape_to_np(shape)

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                cv2.imshow("demo", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


        
if __name__ == '__main__':
    run()
