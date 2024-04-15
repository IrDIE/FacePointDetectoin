import torch
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import albumentations as A
from torchvision import transforms as tfs
from loguru import logger
import shutil
"""
- duplicates in train data
- no data with 2 labeled (!) faces -> if several faces -> only one labeled in data 




"""

def rescale_points(key_pts, old_hw,  output_size):
    h, w = old_hw
    new_h, new_w = int(output_size[0]), int(output_size[1])
    key_pts[:, 0] *= new_w / w
    key_pts[:, 1] *= new_h / h
    return key_pts

def normalise_points(size_in, points, denorm = False):
    # transfer from [0, size_in] to [-1, 1]
    if not denorm :
        points -= size_in // 2
        points /= size_in
    else:
        points *= size_in
        points += size_in // 2
    return points

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, device, train, root_paths : list, labels_ext = 'pts', rescale = 48, transform=None):
        self.labels = [] #[ root_paths + item for root_path in root_paths for item in os.listdir(root_path) if item.split('.')[-1] == labels_ext ]
        self.images = []
        self.device=device
        self.rescale=rescale
        self.train = train
        self.image_transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Resize( (int(self.rescale), int(self.rescale)) ),
            #    tfs.Normalize(),
            ])

        self.augmentation_transform = transform
        for root_path in root_paths:
            for item in os.listdir(root_path):
                if item.split('.')[-1] == labels_ext:
                    self.labels.append( root_path + item)
                else:
                    self.images.append(root_path + item)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        points = load_point(self.labels[idx])
        keypoints = [tuple( [int(point[0]), int(point[1])] ) for point in points] # loaded original-sized keypoints
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_img_shape_ = image.shape
        orig_img_shape = torch.tensor([orig_img_shape_[2], orig_img_shape_[0], orig_img_shape_[1]]) # c hw
        # image and labels to tensor
        if self.train and self.augmentation_transform is not None:
            # label augm
            transformed = self.augmentation_transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']

        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        if self.train :
            keypoints = rescale_points(keypoints, orig_img_shape_[: 2], [self.rescale, self.rescale])
            keypoints = normalise_points(size_in = self.rescale, points=keypoints, denorm=False)
        image = self.image_transform(image)
        if self.train: return image.to(self.device), keypoints.to(self.device)
        else: return image.to(self.device), orig_img_shape, keypoints.to(self.device) # if test : return original-sized points and orig image shape

def load_point(path):
    """takes as input the path to a .pts and returns a list of
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]

    """Use the curly braces to find the start and end of the point data"""
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return points

def get_dataloader(root_paths, device, rescale,train = True, batch_size = None, transform=None):
    if train:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=0.4),
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.3, p=0.1),
            A.RandomSunFlare(flare_roi=(0, 0, 0.8, 0.1), angle_upper=0.1, p=0.1),
            A.HorizontalFlip(p=0.5),
            A.Rotate(p=0.5),

        ], keypoint_params= A.KeypointParams(format='xy', remove_invisible=False))

    dataset = FaceLandmarksDataset(root_paths=root_paths, train=train, device=device,transform = transform, rescale=rescale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader




if __name__ == "__main__":
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    data_pts = '../data/landmarks_task/Menpo/train/aflw__face_39822.pts'
    train_data = [ '../data/landmarks_task/300W/test_crop_crop/',
                  '../data/landmarks_task/300W/test_crop_crop/']
    test_data = [  '../data/landmarks_task/300W/test_crop_crop/',
                  '../data/landmarks_task/300W/test_crop_crop/']



    train_loader = get_dataloader(device=device, train=True, root_paths=train_data, rescale= 48, batch_size=4)
    test_loader = get_dataloader(device=device, train=False, root_paths=test_data, rescale= 48,batch_size=1)
    image, shapes, points = next(iter(test_loader))
    shapes = shapes[0].cpu().detach().numpy()
    # print(f'shapes output = \nimage.shape={image.shape}, \nshapes={list(shapes[1:])}, \n{points.shape}')
    # in test: rescale image and predicted points
    # rescale_points(key_pts, image, output_size)
    #
    image = tfs.Resize( ( int(shapes[1]), int(shapes[2]) ) )(image) #
    frame = np.transpose(image[0].cpu().detach().numpy(), (1, 2, 0))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print(points[0][0])
    points[0] = rescale_points(points[0], ( int(shapes[1]), int(shapes[2]) ) , [100,300])
    print('\nafter\n', points[0][0])

    for i, point in enumerate(points[0]):
        point = list(map(int, point.cpu().detach().numpy()))
        color = (227, 68, 100)
        if i == 30:
            color = (0, 255, 0)
        frame = cv2.circle(frame, point, radius=2, color=color, thickness=-1)

    while True:
        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
