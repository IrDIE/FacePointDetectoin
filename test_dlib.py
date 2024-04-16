import cv2
import math
import os
import dlib
import numpy as np
from utils.dataloaders import load_point
from utils.visualisation import ced_plot

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
#      300 faces In-the-wild challenge: Database and results.
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy


def test_dllib_point_det(predictor_path, datset_path, labels_ext='pts'):
    predictor = dlib.shape_predictor(predictor_path)
    test_errors = []

    for item in os.listdir(datset_path):
        if item.split('.')[-1] != labels_ext:
            points_face = []
            frame = cv2.imread(datset_path + item)
            img_shape = frame.shape
            # print(item, img.shape) # hwc
            label = load_point(datset_path + item.split('.')[-2] + '.pts')
            label = [[int(point[0]), int(point[1])] for point in label]
            det = dlib.rectangle(left=0, top=0, right=img_shape[1], bottom=img_shape[0])
            shape = predictor(frame, det)
            for i in range(68):
                point_x, point_y  = shape.part(i).x, shape.part(i).y
                points_face.append([point_x, point_y])
            mse_per_image = np.square(np.subtract(points_face, label)).mean()
            mse_normed_per_image = mse_per_image / (math.sqrt(int(img_shape[0]) * int(img_shape[1])))

            test_errors.append(mse_normed_per_image)
            # for i, point in enumerate(label):
            #     point = list(map(int, point))
            #     color = (227, 68, 100)
            #     if i == 30:
            #         color = (0, 255, 0)
            #     frame = cv2.circle(frame, point, radius=2, color=color, thickness=-1)
            #
            # while True:
            #     cv2.imshow('', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
    return test_errors


def main():
    predictor_path = 'ckpt/face_dllib_weights/shape_predictor_68_face_landmarks.dat'
    data_name = 'Menpo'
    EXPERIMENT_NAME = 'classic dllib'
    errors_dllib = test_dllib_point_det(predictor_path, datset_path='data/landmarks_task/Menpo/test_clean_crop/')
    errors_dllib = [float(err) for err in errors_dllib]
    ced_plot(error_lists=[errors_dllib], thr_x=0.8, title=f"Cummulative error distribution \nfor data = {data_name}\n{EXPERIMENT_NAME}", save_path='Logs/CED_plots/dllib/')


# if len(sys.argv) != 3:
#     print(
#         "Give the path to the trained shape predictor model as the first "
#         "argument and then the directory containing the facial images.\n"
#         "For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#         "You can download a trained facial shape predictor from:\n"
#         "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#     exit()
#
# predictor_path = sys.argv[1]
# faces_folder_path = sys.argv[2]
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()
#
# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#     print("Processing file: {}".format(f))
#     img = dlib.load_rgb_image(f)
#
#     win.clear_overlay()
#     win.set_image(img)
#
#     # Ask the detector to find the bounding boxes of each face. The 1 in the
#     # second argument indicates that we should upsample the image 1 time. This
#     # will make everything bigger and allow us to detect more faces.
#     dets = detector(img, 1)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#         # Get the landmarks/parts for the face in box d.
#         shape = predictor(img, d)
#         print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
#                                                   shape.part(1)))
#         # Draw the face landmarks on the screen.
#         win.add_overlay(shape)
#
#     win.add_overlay(dets)
#     dlib.hit_enter_to_continue()

if __name__=="__main__":
    main()