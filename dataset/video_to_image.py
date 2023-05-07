'''
    将视频转化为帧
'''
import os
import cv2
from sklearn.model_selection import train_test_split

# 将视频转化为帧
def process_video(video, action_name, save_dir, resize_height, resize_width, root_dir):
    # 将视频处理成图片
    # 获取视频名字
    video_filename = video.split('.')[0]
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.makedirs(os.path.join(save_dir, video_filename))
    # 读视频
    capture = cv2.VideoCapture(os.path.join(root_dir, action_name, video))
    # 读取视频的帧数、高和宽
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))#获取视频帧数
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #确保分割的视频至少有16帧
    EXTRACT_FREQUENCY = 8#隔着4帧取一个
    if frame_count // EXTRACT_FREQUENCY <= 25:
        EXTRACT_FREQUENCY -= 1
        if frame_count // EXTRACT_FREQUENCY <= 25:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 25:
                EXTRACT_FREQUENCY -= 1

    count = 0
    i = 0
    retaining = True
    # 把视频的一帧的高和宽修改成128.171，并命名保存.jpg的图片
    # 保存路径：data/train/视频类别名/一个视频的名字：UCF101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/image_000i.jpg
    while (count < frame_count and retaining):
        retaining, frame = capture.read()#retaining表示是否读出，frame存放到达的内容
        if frame is None:
            continue

        if count % EXTRACT_FREQUENCY == 0:
            if (frame_height != resize_height) or (frame_width != resize_width):
                frame = cv2.resize(frame, (resize_width, resize_height))#修改视频大小
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, 'image_{0:04d}.jpg'.format(i)), img=frame)#保存视频
            # cv2.imwrite(filename=os.path.join(save_dir + "1", video_filename, '0000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1

    # 释放资源
    capture.release()

def v_to_i(root_dir, output_dir, resize_height, resize_width):
    # 创建对应的分组路径
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'train'))
        os.mkdir(os.path.join(output_dir, 'val'))
        os.mkdir(os.path.join(output_dir, 'test'))

    # 划分train/val/test的数据集 0.6/0.2/0.2
    for file in os.listdir(root_dir):
        # 取去每个视频的名字
        video_files = [name for name in os.listdir(os.path.join(root_dir, file))]
        # 划分数据集
        # train&val/test 0.8/0.2
        train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
        # train/val 0.6 .0.2
        train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)
        # 结构为：data/train/视频类别名
        train_dir = os.path.join(output_dir, 'train', file)
        val_dir = os.path.join(output_dir, 'val', file)
        test_dir = os.path.join(output_dir, 'test', file)

        # 判断是否有文件夹，没有创建
        if not os.path.exists(train_dir):
            # windows和linux系统的区别，若在Linux系统下不必添加
            train_dir = train_dir.replace("\\", "/")
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            val_dir = val_dir.replace("\\", "/")
            os.makedirs(val_dir)
        if not os.path.exists(test_dir):
            test_dir = test_dir.replace("\\", "/")
            os.makedirs(test_dir)
        # #把视频转化为图片的形式表示
        for video in train:
            process_video(video, file, train_dir, resize_height, resize_width, root_dir)
        for video in val:
            process_video(video, file, val_dir, resize_height, resize_width, root_dir)
        for video in test:
            process_video(video, file, test_dir, resize_height, resize_width, root_dir)



# 在Pycharm中的输出结果是：
# 总结：
