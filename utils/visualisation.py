import cv2
from dataloaders import load_point
from pathlib import Path


def show_landmarks(file_path, imagename, label_ext ='.pts'):
    """

    :param file_path:
    :param imagename: filename of image
    :return:
    """
    label_name = imagename.split('.')[0] + label_ext
    points = load_point(Path(file_path + label_name))

    frame = cv2.imread(file_path + imagename)

    for i, point in enumerate(points):
        print(i)
        point = list(map(int, point))
        color = (227, 68, 100)
        if i == 30:
            color = (0, 255, 0)
        frame = cv2.circle(frame, point, radius=2, color=color, thickness=-1)

    while True:
        cv2.imshow('',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    data_pts = '../data/landmarks_task/Menpo/train_clean_crop/'
    show_landmarks(file_path=data_pts, imagename='aflw__face_39897.jpg')

