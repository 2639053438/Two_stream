import glob
import os
import torch
import my_spatial_demo_resnet as mm
import video_transforms
from dataset import my_build_of as mb
from models import rgb_resnet as frame_m
from models import vit_model as flow_m
from dataset import my_dataset as md
from my_twostream_net import TwoStreamNet

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # 视频数据集路径,数据集格式：video_path\\视频类别\\视频
    video_path = "D:\\Two_stream\\dataset\\UCF-101"
    # 保存视频帧路径,路径格式：frame_path\\train\\视频类别\\视频名\\图片
    frame_path = "D:\\Two_stream\\UCF-101_frames"
    # 保存视频帧路径,路径格式：flow_path\\train\\视频类别\\视频名\\图片
    flow_path = "D:\\Two_stream\\UCF-101_flows"
    # 图片大小
    resize_height = 256
    resize_width = 256
    # 视频后缀
    ext = 'avi'
    # 图片数据及“路径 标签”txt文件
    # mb.build_image_and_txt(video_path, frame_path, flow_path, resize_height, resize_width, ext)

    # 初始学习率
    lr = 0.001
    # 学习率衰减，为了避免过拟合
    momentum = 0.9
    # 正则化：提高模型的泛化能力，降低过拟合的可能
    weight_decay = 1e-4
    # 采样视频帧的长度
    new_length = 1
    batch_size = 25
    # 线程数
    workers = 8
    start_epoch = 0
    epochs = 400
    # 保存频率
    save_freq = 1
    # 图片类型,可选：['rgb', flow]
    arch = 'flow'
    # 光流图的通道数
    flow_channels = 10
    # 分类的类别数
    num_classes = 101
    # 保存空间网络模型参数路径
    save_frame_pth_path = './checkpoints/frame_pth'
    # 保存时间网络模型参数路径
    save_flow_pth_path = './checkpoints/flow_pth'
    # 保存双流网络模型参数路径
    save_twostream_pth_path = './checkpoints/twostream_pth'
    # 保存模型参数路径
    save_path = [save_frame_pth_path, save_flow_pth_path, save_twostream_pth_path]
    # 创建并训练空间网络
    # mm.two_stream_model(lr=lr, momentum=momentum, weight_decay=weight_decay, new_length=new_length,
    #                     batch_size=batch_size, workers=workers, start_epoch=start_epoch, epochs=epochs,
    #                     save_freq=save_freq, arch='rgb', resize_height=resize_height, resize_width=resize_width,
    #                     num_classes=101, save_path=save_path)
    # # 空间网络最好模型参数
    # frame_pth_best = os.path.join(save_frame_pth_path, "model_best.pth.tar")
    # 创建并训练时间网络
    # mm.two_stream_model(lr=lr, momentum=momentum, weight_decay=weight_decay, new_length=new_length,
    #                     batch_size=batch_size, workers=workers, start_epoch=start_epoch, epochs=epochs,
    #                     save_freq=save_freq, arch='flow', resize_height=resize_height, resize_width=resize_width,
    #                     num_classes=101, save_path=save_path,
    #                     flow_channels=flow_channels)
    # # 时间网络最好模型参数
    # flow_pth_best = os.path.join(save_flow_pth_path, "model_best.pth.tar")

    # 创建并训练双流网络
    mm.two_stream_model(lr=lr, momentum=momentum, weight_decay=weight_decay, new_length=new_length,
                        batch_size=batch_size, workers=workers, start_epoch=start_epoch, epochs=epochs,
                        save_freq=save_freq, arch='twostream', resize_height=resize_height, resize_width=resize_width,
                        num_classes=101, save_path=save_path,
                        flow_channels=flow_channels)
    # 双流网络最好模型参数
    two_stream_pth_best = os.path.join(save_twostream_pth_path, "model_best.pth.tar")

    # 实例化空间网络
    frame_model = frame_m.rgb_resnet152(pretrained=False, num_classes=num_classes)
    frame_model.load_state_dict(frame_pth_best)
    frame_model.cuda()
    frame_model.eval()
    # 实例化时间网络
    flow_model = flow_m.vit_base_patch16_224(num_classes=num_classes, flow_channels=flow_channels * 2)
    flow_model.load_state_dict(flow_pth_best)
    flow_model.cuda()
    flow_model.eval()
    # 进行数据增强
    # Normalize：进行正则化
    clip_mean = [0.485, 0.456, 0.406] * new_length  # new_length：采样视频帧的长度
    clip_std = [0.229, 0.224, 0.225] * new_length
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
    if arch == "rgb":
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
    else:
        scale_ratios = [1.0, 0.875, 0.75]
    test_transform = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.MultiScaleCrop((224, 224), scale_ratios),
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        normalize,
    ])
    # 创建空间testloader
    frame_test_path = "D:\\Two_stream\\dataset\\test_frame_txt.txt"
    frame_test_loader = md.VideoDataset(frame_test_path, batch_size, resize_height=resize_height, resize_width=resize_width, arch="rgb", transform=test_transform)
    # 创建时间testloader
    flow_test_path = "D:\\Two_stream\\dataset\\test_flow_txt.txt"
    flow_test_loader = md.VideoDataset(flow_test_path, batch_size, resize_height=resize_height, resize_width=resize_width, arch="flow", transform=test_transform,
                                       flow_channels=flow_channels)
    # 进行预测
    # 首先看一下在全部测试集上的表现
    # 初始化正确的
    correct = 0
    # 初始化总体的
    total = 0
    with torch.no_grad():
        for data1, data2 in zip(frame_test_loader, flow_test_loader):
            images1, labels1 = data1[0].to(device), data1[1].to(device)
            images2, labels2 = data2[0].to(device), data2[1].to(device)
            outputs1 = frame_model(images1)
            outputs2 = flow_model(images2)
            outputs = (outputs1 + outputs2) / 2
            _, predicted = torch.max(outputs.data, 1)
            total += labels1.size(0)
            correct += (predicted == labels1).sum().item()
    print('Accuracy of the network on the 101 test images: %d %%' % (100 * correct / total))
    # 为了更加细致的看一下模型在哪些类别上表现更好, 在哪些类别上表现更差, 我们分类别的进行准确率计算.
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data1, data2 in zip(frame_test_loader, flow_test_loader):
            images1, labels1 = data1[0].to(device), data1[1].to(device)
            images2, labels2 = data2[0].to(device), data2[1].to(device)
            outputs1 = frame_model(images1)
            outputs1.to(device)
            outputs2 = flow_model(images2)
            outputs2.to(device)
            outputs = (outputs1 + outputs2) / 2
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels1).squeeze()
            for i in range(101):
                label = labels1[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    class_folders = glob.glob(video_path + "\\" + '*')
    classes = []
    for folder in class_folders:
        parts = folder.split('\\')
        classes.append(parts[-1])
    for i in range(num_classes):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
