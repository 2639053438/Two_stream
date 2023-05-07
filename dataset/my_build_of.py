import os
import glob
import argparse
from dataset import video_to_image as vi
from dataset import frames_to_flow as ff


# if __name__ == '__main__':
def build_image_and_txt(video_path="D:\\Two_stream\\dataset\\UCF-101", frame_path="D:\\Two_stream\\UCF-101_frames",
                        flow_path="D:\\Two_stream\\UCF-101_flows", resize_height=224, resize_width=224, ext='avi'):
    out_path_frame = frame_path
    out_path_flow = flow_path
    src_path = video_path
    ext = ext
    resize_height = resize_height
    resize_width = resize_width

    # 创建图片数据集文件夹
    if not os.path.isdir(out_path_frame):
        print("creating folder: " + out_path_frame)
        os.makedirs(out_path_frame)
    if not os.path.isdir(out_path_flow):
        print("creating folder: " + out_path_flow)
        os.makedirs(out_path_flow)

    vid_list = glob.glob(src_path + '/*/*.' + ext)
    print(len(vid_list))
    # 将视频拆成帧
    vi.v_to_i(src_path, out_path_frame, resize_height, resize_width)
    print("finish video to image")
    # 从视频帧中抽取光流图
    ff.f_to_f(out_path_frame, out_path_flow)
    print("finish frame to flow")

    # 构建txt“路径 标签”文件
    for file in os.listdir(out_path_frame):
        frame_txt_path = "D:\\Two_stream\\dataset\\" + file + "_frame_txt.txt"
        frame_txt = open(frame_txt_path, "w")
        i = 0
        for sort_name in os.listdir(os.path.join(out_path_frame, file)):
            i = i + 1
            for video_name in os.listdir(os.path.join(out_path_frame, file, sort_name)):
                frame_paths = os.path.join(out_path_frame, file, sort_name, video_name)
                path = frame_paths + " " + str(i) + '\n'
                frame_txt.write(path)
        frame_txt.close()
    print("finish frame.txt")

    for file in os.listdir(out_path_flow):
        flow_txt_path = "D:\\Two_stream\\dataset\\" + file + "_flow_txt.txt"
        flow_txt = open(flow_txt_path, "w")
        j = 0
        for sort_name in os.listdir(os.path.join(out_path_flow, file)):
            j = j + 1
            for video_name in os.listdir(os.path.join(out_path_flow, file, sort_name)):
                flow_paths = os.path.join(out_path_flow, file, sort_name, video_name)
                path = flow_paths + " " + str(j) + '\n'
                flow_txt.write(path)
        flow_txt.close()
    print("finish flow.txt")











