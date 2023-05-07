import os
import numpy as np
import math
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import models
from VideoSpatialPrediction import VideoSpatialPrediction

def main():
    ##########################################################################################
    #第一部分：测试模型是否可行，运行时间是多少
    # 导入预训练模型参数.pth文件
    model_path = '../../checkpoints/model_best.pth.tar'
    # 数据集路径
    data_dir = "~/UCF101/frames"
    #
    start_frame = 0
    # 分类的类别
    num_categories = 101

    # 获取当前时间
    model_start_time = time.time()
    # 从tar文件中实例化argparse（用来添加预训练参数）
    params = torch.load(model_path)

    #实例化网络，不加载预训练
    spatial_net = models.rgb_resnet152(pretrained=False, num_classes=101)
    # state_dict：字典对象，可用于保存模型参数
    spatial_net.load_state_dict(params['state_dict'])
    # 加载到cuda上
    spatial_net.cuda()
    # 与model.train()相似，唯一不同的是，model.eval()不启用 Batch Normalization（BN层能够用到每一批数据的均值和方差） 和 Dropout（随机取一部分网络连接来训练更新参数）。
    spatial_net.eval()
    # 获取模型结束时间
    model_end_time = time.time()
    # 计算模型运行时间
    model_time = model_end_time - model_start_time
    # 输出模型运行时间
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    ###########################################################################################################
    #第二部分：开始加载数据集，进行预测
    # 加载数据标签txt文件
    val_file = "./testlist01_with_labels.txt"
    f_val = open(val_file, "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))

    # 记录类别
    line_id = 1
    # 记录真确的个数
    match_count = 0
    # 存储分类结果
    result_list = []
    # 获取txt文件中的数据
    for line in val_list:
        line_info = line.split(" ")
        # 获取数据的路径
        clip_path = line_info[0]
        # 获取数据的索引
        input_video_label = int(line_info[1]) - 1

        # 利用空间网络进行分类，得到分类结果spatial_prediction
        spatial_prediction = VideoSpatialPrediction(
                clip_path,#路径
                spatial_net,#网络
                num_categories,#分类数
                start_frame)

        # np.mean：计算给定数组或矩阵的平均值。
        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        # print(avg_spatial_pred_fc8.shape)
        # 保存分类结果
        result_list.append(avg_spatial_pred_fc8)
        # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

        # np.argmax：取最大的值最为分类结果
        pred_index = np.argmax(avg_spatial_pred_fc8)
        print("Sample %d/%d: GT: %d, Prediction: %d" % (line_id, len(val_list), input_video_label, pred_index))

        if pred_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(val_list))
    print("Accuracy is %4.4f" % (float(match_count)/len(val_list)))
    np.save("ucf101_s1_rgb_resnet152.npy", np.array(result_list))

if __name__ == "__main__":
    main()




    # # spatial net prediction
    # class_list = os.listdir(data_dir)
    # class_list.sort()
    # print(class_list)

    # class_index = 0
    # match_count = 0
    # total_clip = 1
    # result_list = []

    # for each_class in class_list:
    #     class_path = os.path.join(data_dir, each_class)

    #     clip_list = os.listdir(class_path)
    #     clip_list.sort()

    #     for each_clip in clip_list:
            # clip_path = os.path.join(class_path, each_clip)
            # spatial_prediction = VideoSpatialPrediction(
            #         clip_path,
            #         spatial_net,
            #         num_categories,
            #         start_frame)

            # avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
            # # print(avg_spatial_pred_fc8.shape)
            # result_list.append(avg_spatial_pred_fc8)
            # # avg_spatial_pred = softmax(avg_spatial_pred_fc8)

            # pred_index = np.argmax(avg_spatial_pred_fc8)
            # print("GT: %d, Prediction: %d" % (class_index, pred_index))

            # if pred_index == class_index:
            #     match_count += 1
#             total_clip += 1

#         class_index += 1

#     print("Accuracy is %4.4f" % (float(match_count)/total_clip))
#     np.save("ucf101_split1_resnet_rgb.npy", np.array(result_list))

# if __name__ == "__main__":
#     main()
