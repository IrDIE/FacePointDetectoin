import sys
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

def get_face(path, show = False):
    def _convert_to_array(rectangles):
        result = []
        for rect in rectangles:
            rect_coord = map(int, [rect.left(), rect.top(), rect.right(), rect.bottom()])
            result.append(list(rect_coord))
        return result

    img = dlib.load_rgb_image(path)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)

    if show:
        print(f"Number of faces detected: {len(dets)}\ndets={dets}")
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()
    else:
        return _convert_to_array(dets), img


if __name__ == "__main__":
    get_face('../data/landmarks_task/Menpo/train_clean/aflw__face_39897.jpg', show=True)

# for f in sys.argv[1:]:
#     print("Processing file: {}".format(f))
#     img = dlib.load_rgb_image(path)
#     # The 1 in the second argument indicates that we should upsample the image
#     # 1 time.  This will make everything bigger and allow us to detect more
#     # faces.
#     dets, scores, idx = detector(img, 1, -1)
#     print(f"Number of faces detected: {len(dets)}\ndets={dets}")
#     for i, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             i, d.left(), d.top(), d.right(), d.bottom()))
#
#     win.clear_overlay()
#     win.set_image(img)
#     win.add_overlay(dets)
#     dlib.hit_enter_to_continue()


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
# if (len(sys.argv[1:]) > 0):
#     img = dlib.load_rgb_image(sys.argv[1])
#     dets, scores, idx = detector.run(img, 1, -1)
#     for i, d in enumerate(dets):
#         print("Detection {}, score: {}, face_type:{}".format(
#             d, scores[i], idx[i]))