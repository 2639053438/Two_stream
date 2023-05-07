import glob
from dataset import guangliu_tu as gl

#视频帧的路径为data\图片类别级\类别级\视频级\图片级

#将图片处理成光流图
def f_to_f(root_path, save_dir):#root_path是图片文件夹的路径，save_dir是保存光流图的路径
# if __name__ == "__main__":
    root_path = 'D:\\Two_stream\\UCF-101_frames'
    save_dir = 'D:\\Two_stream\\UCF-101_flows'
    roots = glob.glob(root_path + "\\" + "*")  # roots获取到的是train级的内容
    for root in roots:
        #root是train级  save_dir是data级
        class_folders = glob.glob(root + "\\" + '*')#class_folders是类别级
        for folder in class_folders:
            videos = glob.glob(folder + '\\' + '*')#videos是视频级
            for video in videos:
                gl.cal_for_frames(video, save_dir)




