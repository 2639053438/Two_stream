import math
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, txt_path='', batch_size=25, resize_height=224, resize_width=224,
                 arch='rgb', flow_channels=10, transform=None, txt_path1='', transform1=None):
        self.flow_channels = flow_channels
        self.arch = arch
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.transform = transform
        self.transform1 = transform1
        self.batch_size = batch_size
        self.frame_paths = []
        self.frame_labels = []
        self.frame_paths1 = []
        self.frame_labels1 = []
        self.txt_path = txt_path
        self.txt_path1 = txt_path1
        txt = open(self.txt_path)
        data = txt.readlines()
        for line in data:
            line_info = line.split()
            frame_path = line_info[0]
            frame_label = str(int(line_info[1]) - 1)
            self.frame_paths.append(frame_path)
            self.frame_labels.append(frame_label)
        # print(self.frame_labels)
        txt.close()
        if arch == 'twostream':
            txt1 = open(self.txt_path1)
            data1 = txt1.readlines()
            for line1 in data1:
                line_info1 = line1.split()
                frame_path1 = line_info1[0]
                frame_label1 = str(int(line_info1[1]) - 1)
                self.frame_paths1.append(frame_path1)
                self.frame_labels1.append(frame_label1)
            # print(self.frame_labels)
            txt1.close()

    def __getitem__(self, index):
        if self.arch == 'rgb':
            # duration总的图片数量
            duration = len(os.listdir(self.frame_paths[index]))
            for i in range(duration):
                # img_file是每张图片的地址:/home/yzhu25/Documents/UCF101/frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/image_000i.jpg
                img_file = os.path.join(self.frame_paths[index], 'image_{0:04d}.jpg'.format(i))
                img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
                img = cv2.resize(img, (self.resize_height, self.resize_width))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                input_data = self.transform(img)
                label = np.array(self.frame_labels[index]).astype(float)
                return input_data, torch.from_numpy(label)
        elif self.arch == 'flow':
            duration = len(os.listdir(self.frame_paths[index]))
            dims = (self.resize_height, self.resize_width, self.flow_channels * 2)
            flow = np.zeros(shape=dims, dtype=np.float64)
            for i in range(int(duration / 2) - self.flow_channels + 1):
                for j in range(self.flow_channels):
                    flow_x_file = os.path.join(self.frame_paths[index], 'flow_x_{0:04d}.jpg'.format(i + j))
                    flow_y_file = os.path.join(self.frame_paths[index], 'flow_y_{0:04d}.jpg'.format(i + j))
                    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                    img_x = cv2.resize(img_x, (self.resize_height, self.resize_width))
                    img_y = cv2.resize(img_y, (self.resize_height, self.resize_width))
                    flow[:, :, j] = img_x
                    flow[:, :, j + 10] = img_y
                flow = self.transform(flow)
                label = np.array(self.frame_labels[index]).astype(float)
                return flow, torch.from_numpy(label)
        else:
            # 随机从当前视频中取一张图片
            rgb_frame_list = os.listdir(self.frame_paths[index])
            randm_rgb = np.random.randint(0, len(rgb_frame_list))
            RGB_image_path = rgb_frame_list[randm_rgb]
            RGB_image_path = os.path.join(self.frame_paths[index], RGB_image_path)
            # print(RGB_image_path)
            img = cv2.imread(RGB_image_path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (self.resize_height, self.resize_width))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_image = self.transform(img)
            # 取光流图
            duration = len(os.listdir(self.frame_paths1[index]))
            dims = (self.resize_height, self.resize_width, self.flow_channels * 2)
            flow = np.zeros(shape=dims, dtype=np.float64)
            for i in range(int(duration / 2) - self.flow_channels + 1):
                for j in range(self.flow_channels):
                    flow_x_file = os.path.join(self.frame_paths1[index], 'flow_x_{0:04d}.jpg'.format(i + j))
                    flow_y_file = os.path.join(self.frame_paths1[index], 'flow_y_{0:04d}.jpg'.format(i + j))
                    img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
                    img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
                    img_x = cv2.resize(img_x, (self.resize_height, self.resize_width))
                    img_y = cv2.resize(img_y, (self.resize_height, self.resize_width))
                    flow[:, :, j] = img_x
                    flow[:, :, j + 10] = img_y
                flow = self.transform1(flow)
                label = np.array(self.frame_labels1[index]).astype(float)
                return flow, torch.from_numpy(label), rgb_image

    def __len__(self):
        return len(self.frame_paths)