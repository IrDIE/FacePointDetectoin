import os, cv2
import shutil
from dataloaders import load_point
from face_detector import get_face
import numpy as np

def clean_data(path, ext_pts = 'pts', n_points = 68 ):
    """
    Get rid of profile faces
    :param path:
    :param ext_pts:
    :param n_points:
    :return:
    """

    clean_dirname = path[:-1] + '_clean/'

    os.makedirs(clean_dirname, exist_ok=True)
    for item in os.listdir(path):
        filename = item.split('.')
        if filename[-1] != ext_pts: # take picture
            ext_image = filename[-1]
            points = load_point(path+filename[-2] + '.' + ext_pts)
            if len(points) % n_points == 0:
                if len(points) > n_points : print(item)
                shutil.copyfile(path + filename[-2] + '.' + ext_pts, clean_dirname + filename[-2] + '.' + ext_pts)
                shutil.copyfile(path + filename[-2] + '.' + ext_image, clean_dirname + filename[-2] + '.' + ext_image)

def save_crop_labels(path, ext_pts='pts'):

    def _if_nose_in_bbox(x_nose, y_nose, bboxs):
        for i, box in enumerate(bboxs):
            if (x_nose < box[2] and x_nose > box[0]) and ( y_nose < box[3] and y_nose > box[1] ):
                return i
            else : return -1

    def _crop(labels, bbox, img):
        """
        :param labels:
        :param bbox: left top right bottom
        """
        max_w_new, max_h_new = bbox[2]-bbox[0], bbox[3]-bbox[1]
        crop = img[bbox[1] : bbox[3],bbox[0] : bbox[2], ::-1]
        new_label = []
        for label in labels:
            x,y = label
            x_new, y_new = x - bbox[0], y - bbox[1]
            x_new = np.clip(x_new, 0, max_w_new - 0.0001)
            y_new = np.clip(y_new, 0, max_h_new - 0.0001)
            new_label.append((x_new, y_new))
        return crop, new_label

    def save_labels(labels, path):
        header_str, tail_str = "version: 1\nn_points: 68\n{\n", "}"
        labels_str = ''


        for label in labels:
            x, y = label
            if x < 0 :  x = 0.
            if y < 0:  y = 0.
            labels_str += str(x)+' '+str(y)+'\n'
        with open(path, "w") as text_file:
            text_file.write(header_str + labels_str + tail_str)

    def extend_dets_rectangle(dets, img, fraction = 0.2):
        h_max, w_max = len(img), len(img[0])

        fraction_w, fraction_h = fraction*(dets[2] - dets[0]), fraction*(dets[3]-dets[1])
        dets[0] = int(np.clip(dets[0] - fraction_w, 0, w_max))
        dets[1] = int(np.clip(dets[1] - fraction_h, 0, h_max))
        dets[2] = int(np.clip(dets[2] + fraction_w, 0, w_max))
        dets[3] = int(np.clip(dets[3] + fraction_h, 0, h_max))

        return dets

    # read image and pass to dllib
    clean_dirname = path[:-1] + '_crop/'
    os.makedirs(clean_dirname, exist_ok=True)
    hash_imgs = []

    for item in os.listdir(path):
        filename = item.split('.')
        if filename[-1] != ext_pts:  # take picture
            ext_image = filename[-1]
            labels = load_point(path + filename[-2] + '.' + ext_pts)
            x_nose, y_nose = labels[30]
            dets, img = get_face(path + item, show=False)

            h, exist = check_hash(img, exist_hashs = hash_imgs)
            hash_imgs.append(h)

            # dets format in [rect.left(), rect.top(), rect.right(), rect.bottom()], img in np array HW

            if len(dets) > 1 and not exist: print(item)
            if len(dets) > 0 and not exist: # lets drop duplicates
                need_bbox_index = _if_nose_in_bbox(x_nose, y_nose, bboxs=dets)
                need_bbox = dets[need_bbox_index]
                need_bbox_extended = extend_dets_rectangle(need_bbox, img)

                crop, new_label = _crop(labels, bbox=need_bbox_extended, img=img)
                cv2.imwrite(f'{clean_dirname}{item}', crop)
                save_labels(new_label, path = f'{clean_dirname}'+filename[-2] + '.' + ext_pts)


def check_hash(img, exist_hashs):
    h = hash(str(img))
    return h, h in exist_hashs


if __name__ == "__main__":
    data_pts = '../data/landmarks_task/Menpo/train/aflw__face_39822.pts'
    data = '../data/landmarks_task/300W/test_crop/'

    all_4_data = ['../data/landmarks_task/Menpo/train_clean/',
                  '../data/landmarks_task/Menpo/test_clean/',
                  '../data/landmarks_task/300W/train_clean/',
                  '../data/landmarks_task/300W/test_clean/']
    # clean_data('../data/landmarks_task/300W/test_clean/')
    # save_crop_labels(path=data)

    for data in all_4_data:
        print(f'\n{data}\n***\n')
        save_crop_labels(path = data)