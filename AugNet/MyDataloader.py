#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from glob import glob
from torch.utils.data.dataset import Dataset
import random
from torchvision.transforms import transforms
import torch
from PIL import Image


class MyDatasetSingle(Dataset):
    def __init__(self, img_dir,partition, transform=None):
        random.seed(2023)
        # print(stego_dir)
        self.transform = transform
        self.img_dir = img_dir

        # self.imgs_list_all = [x.split('/')[-1] for x in glob(self.img_dir + '/*')] #仅ubuntu使用
        self.imgs_list_all = [os.path.basename(x) for x in glob(os.path.join(self.img_dir, '*'))]
        random.shuffle(self.imgs_list_all)
        if (partition == 0):
            self.img_list = self.imgs_list_all[:5000]
            self.img_paths= [os.path.join(self.img_dir, x) for x in  self.img_list]

        if (partition == 1):
            self.img_list = self.imgs_list_all[5000:6000]
            self.img_paths = [os.path.join(self.img_dir, x) for x in self.img_list]
        if (partition == 2):
            self.img_list = self.imgs_list_all[:10000]
            self.img_paths = [os.path.join(self.img_dir, x) for x in self.img_list]

        assert len(self.img_paths) != 0, "img_dir is empty"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        file_index = int(idx)

        img_path = self.img_paths[file_index]
        name = os.path.basename(img_path)
        img_data = cv2.imread(img_path, -1)#用图片的原来的格式打开
        img_data = np.expand_dims(img_data, axis=0)
        # print(img_data.shape)
        label = np.array([0], dtype='int32')

        sample = {'data': img_data, 'label': label, 'name':name}
        if self.transform:
            sample = self.transform(sample)
        return sample
