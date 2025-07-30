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


class MyDataset(Dataset):
    def __init__(self, cover_dir, stego_dir,partition, transform=None):
        random.seed(2023)
        # print(stego_dir)
        self.transform = transform

        self.cover_dir = cover_dir
        self.stego_dir = stego_dir

        # self.covers_list_all = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')] #仅ubuntu使用
        self.covers_list_all = [os.path.basename(x) for x in glob(os.path.join(self.cover_dir, '*'))]
        # print(self.covers_list_all)
        # random.shuffle(self.covers_list_all)
        if (partition == 0):
            self.cover_list = self.covers_list_all[:7000]
            self.cover_paths= [os.path.join(self.cover_dir, x) for x in  self.cover_list]
            # self.cover_paths_2 = [os.path.join(self.cover_dir_2, x) for x in self.cover_list]

            self.cover_paths = self.cover_paths 
            #print (self.cover_paths_3)
            #print self.cover_paths
            # print(self.cover_list)
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]
            # self.stego_paths_2 = [os.path.join(self.stego_dir_2, x) for x in self.cover_list]

            self.stego_paths = self.stego_paths
            # print(self.stego_paths)
            self.cover_steg = list(zip(self.cover_paths, self.stego_paths))
            # print(self.cover_steg)
            random.shuffle(self.cover_steg)
            self.cover_paths, self.stego_paths = zip(*self.cover_steg)


        if (partition == 1):
            self.cover_list = self.covers_list_all[7000:8000]
            self.cover_paths = [os.path.join(self.cover_dir, x) for x in self.cover_list]
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]
        if (partition == 2):
            self.cover_list = self.covers_list_all[8000:10000]
            self.cover_paths = [os.path.join(self.cover_dir, x) for x in self.cover_list]
            self.stego_paths = [os.path.join(self.stego_dir, x) for x in self.cover_list]

        assert len(self.cover_paths) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_paths)

    def __getitem__(self, idx):
        file_index = int(idx)

        cover_path = self.cover_paths[file_index]
        stego_path = self.stego_paths[file_index]
        cover_data = cv2.imread(cover_path, -1)#用图片的原来的格式打开
        stego_data = cv2.imread(stego_path, -1)
        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

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

class MyDatasetPair(Dataset):
    def __init__(self, txt, Type='train'):
        self.imgs = []
        self.Type = Type

        # 读取图像路径和标签
        with open(txt, 'r') as fh:
            for line in fh:
                line = line.rstrip()
                words = line.split()

                if os.path.exists(words[0]):
                    self.imgs.append((words[0], int(words[1])))
                else:
                    print(f"File not found: {words[0]}")

        # 将奇数和偶数位置的图像配对，并置乱顺序
        self._pair_and_shuffle()
        if Type=='train':
            self.transform = transforms.Compose([
                AugDataPair(),  # AugData() is the D4 augmentation (Baseline)
                ToTensorPair()
            ])
        else:
            self.transform = transforms.Compose([
                ToTensorPair()
            ])

    def _pair_and_shuffle(self):
        # 将图像路径分为奇数和偶数位置两部分
        even_imgs = self.imgs[::2]  # 偶数位置
        odd_imgs = self.imgs[1::2]  # 奇数位置

        # 判断两个列表是否长度一致
        if len(even_imgs) != len(odd_imgs):
            min_len = min(len(even_imgs), len(odd_imgs))
            even_imgs = even_imgs[:min_len]
            odd_imgs = odd_imgs[:min_len]

        # 将偶数和奇数位置的图像配对，并置乱顺序
        # self.cover = even_imgs
        # self.stego = odd_imgs
        self.cover_steg = list(zip(even_imgs, odd_imgs))
        # print("self.cover_steg:",self.cover_steg)
        random.shuffle(self.cover_steg)

    def __len__(self):
        return len(self.cover_steg)

    def __getitem__(self, index):
        (cover_path, cover_label), (stego_path, stego_label) = self.cover_steg[index]
        # print("cover_path:", cover_path)
        # print("stego_path:", stego_path)
        name = os.path.basename(stego_path)
        # 打开图像并进行转换
        with Image.open(cover_path) as cover_img, Image.open(stego_path) as stego_img:
            sample = {'cover':  np.array(cover_img), 'stego': np.array(stego_img),'name':name}
            # print("cover.shape:",sample['cover'].shape)[256,256]
            if self.transform:
                sample = self.transform(sample)
        return sample

# Data augmentation
class AugDataPair():
    def __call__(self, sample):
        cover, stego = sample['cover'], sample['stego']
        # print("data.shape1:",data.shape)
        # Rotation
        rot = random.randint(0, 3)
        cover = np.rot90(cover, rot, axes=[0,1]).copy()
        stego = np.rot90(stego, rot, axes=[0,1]).copy()

        # Mirroring
        if random.random() < 0.5:
            cover = np.flip(cover, axis=1).copy()
            stego = np.flip(stego, axis=1).copy()
        new_sample = {'cover': cover, 'stego': stego,'name': sample['name']}

        return new_sample


class ToTensorPair():
    def __call__(self, sample):
        cover, stego = sample['cover'], sample['stego']
        noise = stego.astype(np.int16) - cover.astype(np.int16)#有正有负
        cover = np.expand_dims(cover, axis=0)
        noise = np.expand_dims(noise, axis=0)
        # print("stego.shape:", stego.shape) #(1, 256, 256)
        cover = cover.astype(np.float32)
        noise = noise.astype(np.float32)
        # data = data / 255.0

        new_sample = {
            'cover': torch.from_numpy(cover),
            'noise': torch.from_numpy(noise),
            'name': sample['name']
        }
        return new_sample