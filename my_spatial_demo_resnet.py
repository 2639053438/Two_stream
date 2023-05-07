import collections
import shutil
import time
from models import vit_model as vm
import models
import os
import torch
import torch.nn as nn
import video_transforms
from dataset import my_dataset as md
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from my_twostream_net import TwoStreamNet

def adjust_learning_rate(optimizer, epoch):
    """#将学习率设置为初始LR每150个周期衰减10"""
    lr = 0.001
    lr = lr * (0.1 ** (epoch // 150))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """为k的指定值计算precision@k"""
    maxk = max(topk) # 3
    batch_size = target.size(0) #25

    # pred是选出概率最大的标签（可以是一个，也可以是k个）
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch, arch):
    batch_time = AverageMeter()#AverageMeter：计算并存储平均值和当前值
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # 切换到训练模式
    model.train()

    #获取当前时间end
    end = time.time()
    if arch != "twostream":
        for i, (input, target) in enumerate(train_loader):
            # 测量数据加载时间
            data_time.update(time.time() - end)
            # print(input.shape)
            # print(target.shape)
            input = input.float().cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var.long())
            # print(torch.isnan(loss).any())
            # print(torch.isinf(loss).any())
            # print(loss.data.item())
            # print(input.size(0))
            # 测量精度并记录损耗
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            # losses.update(loss.data[0], input.size(0))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # 计算梯度，做SGD步长
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 测量运行时间
            batch_time.update(time.time() - end)
            end = time.time()

            #print-freq：打印频率
            print_freq = 20
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top3=top3))
        return losses, top1
    else:
        for i, (input_flow, target, input_frame) in enumerate(train_loader):
            # 测量数据加载时间
            data_time.update(time.time() - end)
            # print(input.shape)
            # print(target.shape)
            input_flow = input_flow.float().cuda()
            input_frame = input_frame.float().cuda()
            target = target.cuda()
            input_var_flow = torch.autograd.Variable(input_flow)
            input_var_frame = torch.autograd.Variable(input_frame)
            target_var = torch.autograd.Variable(target)

            output = model(input_var_frame, input_var_flow)
            loss = criterion(output, target_var.long())
            # print(torch.isnan(loss).any())
            # print(torch.isinf(loss).any())
            # print(loss.data.item())
            # print(input.size(0))
            # 测量精度并记录损耗
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            # losses.update(loss.data[0], input.size(0))
            losses.update(loss.data.item(), input_flow.size(0))
            top1.update(prec1.item(), input_flow.size(0))
            top3.update(prec3.item(), input_flow.size(0))

            # 计算梯度，做SGD步长
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 测量运行时间
            batch_time.update(time.time() - end)
            end = time.time()

            # print-freq：打印频率
            print_freq = 20
            if i % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top3=top3))
        return losses, top1


def validate(val_loader, model, criterion, arch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # 切换到评估模式
    model.eval()

    end = time.time()
    if arch != "twostream":
        for i, (input, target) in enumerate(val_loader):
            input = input.float().cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # 计算输出
            output = model(input_var)
            loss = criterion(output, target_var.long())

            # 测量精度并记录损耗
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # 测量运行时间
            batch_time.update(time.time() - end)
            end = time.time()

            print_freq = 20
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))

        return top1.avg, losses
    else:
        for i, (input_flow, target, input_frame) in enumerate(val_loader):
            input_flow = input_flow.float().cuda()
            input_frame = input_frame.float().cuda()
            target = target.cuda()
            input_var_flow = torch.autograd.Variable(input_flow, volatile=True)
            input_var_frame = torch.autograd.Variable(input_frame, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # 计算输出
            output = model(input_var_frame, input_var_flow)
            loss = criterion(output, target_var.long())

            # 测量精度并记录损耗
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data.item(), input_flow.size(0))
            top1.update(prec1.item(), input_flow.size(0))
            top3.update(prec3.item(), input_flow.size(0))

            # 测量运行时间
            batch_time.update(time.time() - end)
            end = time.time()

            print_freq = 20
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top3=top3))

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))

        return top1.avg, losses

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)


def change_key_names(old_params, in_channels):
    #  原来的是：(25, 3, w, h)   现在的是：(25, 20, w, h)
    new_params = collections.OrderedDict()
    layer_count = 0
    allKeyList = old_params.keys()
    print(allKeyList)
    for layer_key in allKeyList:
        if layer_count == 2:
            rgb_weight = old_params[layer_key]
            rgb_weight_mean = torch.mean(rgb_weight, dim=1)
            flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1, in_channels, 1, 1)
            new_params[layer_key] = flow_weight
            layer_count += 1
        # if layer_count == 3:
        #     rgb_weight = old_params[layer_key]
        #     rgb_weight_mean = torch.mean(rgb_weight, dim=1)
        #     flow_weight = rgb_weight_mean.unsqueeze(1).repeat(1, in_channels, 1, 1)
        #     new_params[layer_key] = flow_weight
        #     layer_count += 1
        else:
            new_params[layer_key] = old_params[layer_key]
            layer_count += 1

    return new_params

# if __name__ == "__main__":
def two_stream_model(lr=0.001, momentum=0.9, weight_decay=1e-4, new_length=1, batch_size=25, workers=1,
                     start_epoch=0, epochs=400, save_freq=1, arch='flow', resize_height=224, resize_width=224,
                     flow_channels=5, num_classes=101, save_path=['./checkpoints/frame_pth', './checkpoints/flow_pth']):
    # 第一部分：加载与训练参数，进行训练
    # 参数设置
    lr = lr           # 初始学习率
    momentum = momentum       # 学习率衰减，为了避免过拟合
    weight_decay = weight_decay  # 正则化：提高模型的泛化能力，降低过拟合的可能,做的是减小方差.
    new_length = new_length       # 采样视频帧的长度
    batch_size = batch_size
    workers = workers          # 线程数
    start_epoch = start_epoch
    epochs = epochs
    save_freq = save_freq        # 保存频率
    arch = arch
    save_path = save_path
    if not os.path.exists(save_path[0]):
        os.makedirs(save_path[0])
    if not os.path.exists(save_path[1]):
        os.makedirs(save_path[1])
    if not os.path.exists(save_path[2]):
        os.makedirs(save_path[2])
    if arch == 'rgb':
        resume = save_path[0]  # 保存字典路径
    elif arch == 'frame':
        resume = save_path[1]
    else:
        resume = save_path[2]
    resize_height, resize_width = resize_height, resize_width
    flow_channels = flow_channels
    num_classes = num_classes
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    # 创建模型
    if arch == 'rgb':
        model = models.__dict__["rgb_resnet152"](pretrained=True, num_classes=101)
    elif arch== 'flow':
        model = vm.vit_base_patch16_224(num_classes=num_classes, flow_channels=flow_channels * 2)
        model_dict = model.state_dict()
        weights = 'D:\\Two_stream\\models\\vit_base_patch16_224.pth'
        if weights != "":
            assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
            # 读取预训练模型
            weights_dict = torch.load(weights, map_location=device)
            del_keys = ['head.weight', 'head.bias'] if not model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            new_weights_dict = change_key_names(weights_dict, 2 * flow_channels)
            new_weights_dict = {k: v for k, v in new_weights_dict.items() if k in model_dict}
            model_dict.update(new_weights_dict)
            model.load_state_dict(model_dict, strict=False)
        # model = models.__dict__["flow_resnet152"](pretrained=True, num_classes=101)
    else:
        model = TwoStreamNet()

    # DataParallel：将把batch_size划分并分配给所有可用的gpu
    model = torch.nn.DataParallel(model).cuda()
    # # 创建最新检查路径
    # if not os.path.exists(resume):
    #     os.makedirs(resume)
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    # model.parameters()：是PyTorch中模型参数的迭代器。它返回一个生成器，可以用来遍历模型中所有可学习的参数。每一次调用这个函数返回的都是一个参数张量。
    optimizer = torch.optim.SGD(model.parameters(), lr,     # 初始学习率
                                momentum=momentum,          # momentum：学习率衰减，为了避免过拟合
                                weight_decay=weight_decay)  # weight_decay：正则化：提高模型的泛化能力，降低过拟合的可能。做的是减小方差
    # 数据转换
    # Normalize：进行正则化
    clip_mean = [0.485, 0.456, 0.406] * new_length     # new_length：采样视频帧的长度
    clip_std = [0.229, 0.224, 0.225] * new_length
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)

    if arch == "rgb":
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
    elif arch == "flow":
        scale_ratios = [1.0, 0.875, 0.75]
    else:
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        scale_ratios1 = [1.0, 0.875, 0.75]

    # Compose：将几个video_transform组合在一起。
    # Scale：将输入numpy数组重新调整为给定的“大小”。if h > w，(h, w)->(s * h / 2, s)
    # MultiScaleCrop:数据增强，第一个参数为网络输入所需要的高度和宽度，scale_ratios是有效的抖动，具体看http://arxiv.org/abs/1507.02159
    # RandomHorizontalFlip：随机水平翻转给定的numpy数组，概率为0.5
    # ToTensor()：将numpy转为tensor
    train_transform = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.MultiScaleCrop((224, 224), scale_ratios),  #
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        normalize,
    ])
    # CenterCrop：在中间裁剪给定numpy数组，使其具有给定大小的区域。
    val_transform = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.CenterCrop((224)),
        video_transforms.ToTensor(),
        normalize,
    ])
    train_transform1 = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.MultiScaleCrop((224, 224), scale_ratios1),  #
        video_transforms.RandomHorizontalFlip(),
        video_transforms.ToTensor(),
        normalize,
    ])
    # CenterCrop：在中间裁剪给定numpy数组，使其具有给定大小的区域。
    val_transform1 = video_transforms.Compose([
        # video_transforms.Scale((256)),
        video_transforms.CenterCrop((224)),
        video_transforms.ToTensor(),
        normalize,
    ])
    # 数据加载
    # 文件格式:./datasets/settings/ucf101/train_rgb_split1.txt
    if arch == "rgb":
        train_split_file = "D:\\Two_stream\\dataset\\train_frame_txt.txt"
        train_dataset = md.VideoDataset(train_split_file, batch_size, resize_height, resize_width, arch,
                                        transform=train_transform)
    elif arch == "flow":
        train_split_file = "D:\\Two_stream\\dataset\\train_flow_txt.txt"
        train_dataset = md.VideoDataset(train_split_file, batch_size, resize_height, resize_width, arch,
                                        flow_channels=flow_channels, transform=train_transform)
    else:
        train_split_file = "D:\\Two_stream\\dataset\\train_flow_txt.txt"
        train_split_file1 = "D:\\Two_stream\\dataset\\train_flow_txt.txt"
        train_dataset = md.VideoDataset(train_split_file, batch_size, resize_height, resize_width, arch,
                                        flow_channels=flow_channels, transform=train_transform,
                                        txt_path1=train_split_file1, transform1=train_transform1)
    if arch == "rgb":
        val_split_file = "D:\\Two_stream\\dataset\\val_frame_txt.txt"
        val_dataset = md.VideoDataset(val_split_file, batch_size, resize_height, resize_width, arch,
                                      transform=val_transform)
    elif arch == "flow":
        val_split_file = "D:\\Two_stream\\dataset\\val_flow_txt.txt"
        val_dataset = md.VideoDataset(val_split_file, batch_size, resize_height, resize_width, arch,
                                      flow_channels=flow_channels, transform=val_transform)
    else:
        val_split_file = "D:\\Two_stream\\dataset\\val_frame_txt.txt"
        val_split_file1 = "D:\\Two_stream\\dataset\\val_flow_txt.txt"
        val_dataset = md.VideoDataset(val_split_file, batch_size, resize_height, resize_width, arch,
                                      transform=val_transform,
                                      txt_path1=val_split_file1, transform1=val_transform1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,  # shuffle：Ture为打乱数据，False为不打乱
        num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=True,  # shuffle：Ture为打乱数据，False为不打乱
        num_workers=workers, pin_memory=True)
    if arch == "rgb":
        summarywriter_path = "D:\\Two_stream\\runs\\frame"
    elif arch == "flow":
        summarywriter_path = "D:\\Two_stream\\runs\\flow"
    else:
        summarywriter_path = "D:\\Two_stream\\runs\\twostream"

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)# 将学习率设置为初始LR每150个周期衰减10

        # 一次训练
        losses, top1 = train(train_loader, model, criterion, optimizer, epoch, arch)
        writer = SummaryWriter(summarywriter_path)
        writer.add_scalar("train_loss", losses.avg, epoch)
        writer.add_scalar("train_acc", top1.avg, epoch)
        # 在验证集上求值
        prec1, loss = validate(val_loader, model, criterion, arch)
        writer.add_scalar("val_loss", loss.avg, epoch)
        writer.add_scalar("val_acc", prec1, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.close()
        # 记住最好prec@l并保存检查点
        best_prec1 = 0
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if (epoch + 1) % save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({#保存模型参数
                'epoch': epoch + 1,
                'arch': arch,#图像类型
                'state_dict': model.state_dict(),#state_dict()：返回一个包含整个模块状态的字典。
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_name, resume)#resume：保存模型字典的路径