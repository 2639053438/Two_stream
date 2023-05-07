'''
A sample function for classification using spatial network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import os
import sys
import numpy as np
import math
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../../")
import video_transforms

def VideoSpatialPrediction(
        vid_name,#/home/yzhu25/Documents/UCF101/frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g04_c03
        net,
        num_categories,#类别
        start_frame=0,
        num_frames=0,
        num_samples=25#num_samples 是每次输入图片的帧数
        ):

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        duration = len(imglist)
        # print(duration)
    else:
        duration = num_frames

    # Normalize：进行正则化
    clip_mean = [0.485, 0.456, 0.406]
    clip_std = [0.229, 0.224, 0.225]
    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    # 选择
    #取图片的步长
    step = int(math.floor((duration-1)/(num_samples-1)))
    dims = (256,340,3,num_samples)
    # 创建输入矩阵
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        # img_file是每张图片的地址:/home/yzhu25/Documents/UCF101/frames/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/image_000i.jpg
        img_file = os.path.join(vid_name, 'image_{0:04d}.jpg'.format(i*step+1))
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dims[1::-1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb[:,:,:,i] = img
        rgb_flip[:,:,:,i] = img[:,::-1,:]# img[:,::-1,:]水平翻转

    # 对一张图片进行裁剪拼接
    # 将rgb图片的w和h从前往后：0到224裁剪
    rgb_1 = rgb[:224, :224, :,:]
    # 将rgb图片的w从前往后：0到224裁剪，h从后往前：-224到0
    rgb_2 = rgb[:224, -224:, :,:]
    rgb_3 = rgb[16:240, 60:284, :,:]
    rgb_4 = rgb[-224:, :224, :,:]
    rgb_5 = rgb[-224:, -224:, :,:]
    rgb_f_1 = rgb_flip[:224, :224, :,:]
    rgb_f_2 = rgb_flip[:224, -224:, :,:]
    rgb_f_3 = rgb_flip[16:240, 60:284, :,:]
    rgb_f_4 = rgb_flip[-224:, :224, :,:]
    rgb_f_5 = rgb_flip[-224:, -224:, :,:]

    rgb = np.concatenate((rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5), axis=3)

    _, _, _, c = rgb.shape
    rgb_list = []
    for c_index in range(c):
        # 将rbg中的图片取出
        cur_img = rgb[:,:,:,c_index].squeeze()# squeeze()函数智能压缩维度为1的矩阵
        # 将图片正则化
        cur_img_tensor = val_transform(cur_img)
        # .numpy()将tensor转换为numpy格式
        # expand_dims：插入一个新轴
        # 将图片恢复为类似rbg格式：(1, 256, 340, 3)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))

    #将 rgb_list中的内容合并成一个块：(25, 256, 340, 3)
    rgb_np = np.concatenate(rgb_list,axis=0)
    # print(rgb_np.shape)
    batch_size = 25
    # 创建一个类别*每个输入数据rgb中图片的个数的参数：101*25
    prediction = np.zeros((num_categories,rgb.shape[3]))
    # ceil：上限
    num_batches = int(math.ceil(float(rgb.shape[3])/batch_size))

    for bb in range(num_batches):
        span = range(batch_size*bb, min(rgb.shape[3],batch_size*(bb+1)))
        input_data = rgb_np[span,:,:,:]
        ##########################################################################################
        imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
        # torch.autograd.Variable:将Tensor转换为Variable，可以装载梯度信息
        imgDataVar = torch.autograd.Variable(imgDataTensor)
        #进入网络的输入的格式：(25, 256, 340, 3)
        output = net(imgDataVar)
        result = output.data.cpu().numpy()
        #np.transpose反转或置换一个数组的轴
        # 得到每个图片的分类结果
        prediction[:, span] = np.transpose(result)

    return prediction
