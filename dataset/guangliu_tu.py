import os
import numpy as np
import cv2
import glob

from matplotlib import pyplot as plt

_IMAGE_SIZE = 256


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def cal_for_frames(video_path,save_dir):#video_path是视频级，save_dir是视频级
    frames = glob.glob(os.path.join(video_path, '*.jpg'))#frames中是图片级
    # print(frames)
    frames.sort()#sort()是对地址进行排序

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(prev)
    hsv[..., 1] = 255
    count = 0
    # TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    for frame_curr in frames[1:]:
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, pyr_scale=0.5, levels=3, winsize=55,
                                            iterations=3, poly_n=7, poly_sigma=1.5,
                                            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow_img = flow_to_image(flow)
        parts = video_path.split('\\')
        filename = '\\' + parts[-3] + '\\' + parts[-2] + '\\' + parts[-1]
        # label_dir = filename
        save_dir_final = save_dir + filename
        if not os.path.exists(save_dir_final):
            os.makedirs(save_dir_final)
        # cv2.imwrite(save_dir_final + "\\frame{0:06d}.jpg".format(count), flow_img)

        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)

        horz = horz.astype('uint8')
        vert = vert.astype('uint8')

        cv2.imwrite(save_dir_final + "\\flow_x_{0:04d}.jpg".format(count), horz)
        cv2.imwrite(save_dir_final + "\\flow_y_{0:04d}.jpg".format(count), vert)

        count += 1
        prev = curr
    return flow
