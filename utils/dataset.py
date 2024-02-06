import os
from os.path import splitext
from os import listdir
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import torchvision.transforms.functional as transF
from imgaug import augmenters as iaa
import scipy.io as io
from osgeo import gdal

mean_std_dict = {
    'WHU': ['WHU', [0.43526826, 0.44523221, 0.41307611], [0.20436029, 0.19237618, 0.20128716], '.tif'],
    'Mass': ['Mass', [0.32208377, 0.32742606, 0.2946236], [0.18352227, 0.17701593, 0.18039343], '.tif'],
    'Inria': ['Inria', [0.42314604, 0.43858219, 0.40343547], [0.18447358, 0.16981384, 0.1629876], '.tif']
}


class BuildingDataset(Dataset):
    def __init__(self, dataset_dir, training=False, txt_name: str = "train.txt", data_name='WHU'):
        self.name, self.mean, self.std, self.shuffix = mean_std_dict[data_name]
        if self.name == 'Mass':
            self.imgs_dir = os.path.join(dataset_dir, 'TIFFImages')
            self.labels_dir = os.path.join(dataset_dir, 'SegmentationClass')
            self.dis_dir = os.path.join(dataset_dir, 'boundary')
            txt_path = os.path.join(dataset_dir, "ImageSets", "Segmentation", txt_name)
            assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
            with open(os.path.join(txt_path), "r") as f:
                file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.scale = 1
            self.training = training
            self.name, self.mean, self.std, self.shuffix = mean_std_dict[data_name]
            self.images = [os.path.join(self.imgs_dir, x + self.shuffix) for x in file_names]
            self.labels = [os.path.join(self.labels_dir, x + self.shuffix) for x in file_names]
            self.dis = [os.path.join(self.dis_dir, x + '.mat') for x in file_names]
            assert (len(self.images) == len(self.labels)) & (len(self.images) == len(self.dis))

            logging.info(f'Creating dataset with {len(self.images)} examples')

        else:
            mode = txt_name.split(".")[0]
            self.imgs_dir = os.path.join(dataset_dir, mode, 'image')
            self.labels_dir = os.path.join(dataset_dir, mode, 'label')
            self.dis_dir = os.path.join(dataset_dir, 'boundary')
            txt_path = os.path.join(dataset_dir, "dataset", txt_name)
            assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
            with open(os.path.join(txt_path), "r") as f:
                file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
            self.scale = 1
            self.training = training
            self.images = [os.path.join(self.imgs_dir, x + self.shuffix) for x in file_names]
            self.labels = [os.path.join(self.labels_dir, x + self.shuffix) for x in file_names]

            self.dis = [os.path.join(self.dis_dir, x + '.mat') for x in file_names]
            assert (len(self.images) == len(self.labels)) & (len(self.images) == len(self.dis))

            logging.info(f'Creating dataset with {len(self.images)} examples')

        # 影像预处理方法
        self.transform = iaa.Sequential([
            iaa.Rot90([0, 1, 2, 3]),
            iaa.VerticalFlip(p=0.5),
            iaa.HorizontalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.images)

    def _load_mat(self, filename):
        return io.loadmat(filename)

    def _load_maps(self, filename, ):
        dct = self._load_mat(filename)
        distance_map = dct['depth'].astype(np.int32)
        return distance_map

    def __getitem__(self, index):
        if self.name == 'Mass':
            img_file = self.images[index]
            img = np.array(Image.open(img_file))
            labels = readTif(self.labels[index])
            width = labels.RasterXSize
            height = labels.RasterYSize
            label = labels.ReadAsArray(0, 0, width, height)
            label = label[0, :, :] / 255
        elif self.name == 'WHU':
            img_file = self.images[index]
            img = np.array(Image.open(img_file))
            label_file = self.labels[index]
            label = np.array(Image.open(label_file).convert("P")).astype(np.int16) / 255.
        elif self.name == 'Inria':
            img_file = self.images[index]
            img = np.array(Image.open(img_file))
            label_file = self.labels[index]
            label = np.array(Image.open(label_file).convert("P")).astype(np.int16) / 255.

        # 利用_load_maps获取得到的distance_map和angle_map
        if self.training:
            distance_map = self._load_maps(self.dis[index])
            distance_map = np.array(distance_map)
            img, label = self.transform(image=img, segmentation_maps=np.stack(
                (label[np.newaxis, :, :], distance_map[np.newaxis, :, :]), axis=-1).astype(np.int32))

            label, distance_map = label[0, :, :, 0], label[0, :, :, 1]

        img, label = transF.to_tensor(img.copy()), (transF.to_tensor(label.copy()) > 0).int()
        # 标准化
        img = transF.normalize(img, self.mean, self.std)
        if self.training:
            return {
                'image': img.float(),
                'label': label.float(),
                'distance_map': distance_map,
                'name': self.images[index]
            }
        else:
            return {
                'image': img.float(),
                'label': label.float(),
                'name': self.images[index]
            }


def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "can not open the file")
    return dataset
