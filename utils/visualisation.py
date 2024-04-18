import cv2
from .dataloaders import load_point
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger
import matplotlib.patches as mpatches
from random import random

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
def ced_plot(error_lists, save_path, data_name, title = '', thr_x = 0.08):
    os.makedirs(save_path, exist_ok=True)
    legend = []
    plt.figure(figsize=(10, 10), dpi=80)

    for error_list in error_lists:
        color = (random(), random(), random())
        values, base = np.histogram(error_list, bins=len(error_list))
        cumulative = np.cumsum(values) / len(values)
        print(cumulative)
        print(base)
        plt.plot(base[:-1], cumulative, color=color)
        area = np.round(np.trapz(y=cumulative[base[1:] <= thr_x], x=base[base <= thr_x][1:] ), 7)
        legend.append(mpatches.Patch(color=color, label=f'\nArea={area}'))

    plt.legend(handles=legend, fontsize=18)
    plt.title(title, fontsize=18)
    plt.grid()
    plt.ylabel('Fraction of test imeges with error < CED ', fontsize=18)
    plt.xlabel('Normalised mean error', fontsize=18)
    plt.xlim([0, thr_x])
    plt.ylim([0, 1])

    plt.savefig(f'{save_path}ced_{data_name}.png')
    logger.info(f'CED plot saved to: {save_path}')


if __name__ == "__main__":

    data_pts = '../data/landmarks_task/300W/train_clean_crop/'
    show_landmarks(file_path=data_pts, imagename='134212_1.jpg') # 167629013_1

