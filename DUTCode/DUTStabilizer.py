import torch
import torch.nn as nn
import numpy as np
import os
import math
import cv2
import sys

# # 获取当前脚本所在的目录路径
# current_dir = os.path.dirname(__file__)
# print(current_dir)
# 获取 models 文件夹所在的路径
# models_dir = os.path.abspath(os.path.join(current_dir, "models"))
# print(models_dir)

# parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
# # print('parentddir = ' + parentddir)
# sys.path.append(parentddir)

from DUTCode.models.DUT.DUT import DUT
from tqdm import tqdm
from utils.WarpUtils import warpListImage
from configs.config import cfg
import argparse

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="Control for stabilization model")
    parser.add_argument(
        "--SmootherPath",
        help="the path to pretrained smoother model, blank for jacobi solver",
        default="",
    )
    parser.add_argument(
        "--RFDetPath",
        help="pretrained RFNet path, blank for corner detection",
        default="",
    )
    parser.add_argument(
        "--PWCNetPath", help="pretrained pwcnet path, blank for KTL tracker", default=""
    )
    parser.add_argument(
        "--MotionProPath",
        help="pretrained motion propagation model path, blank for median",
        default="",
    )
    parser.add_argument(
        "--SingleHomo",
        help="whether use multi homograph to do motion estimation",
        action="store_true",
    )
    parser.add_argument(
        "--InputBasePath", help="path to input videos (cliped as frames)", default=""
    )
    parser.add_argument(
        "--OutputBasePath", help="path to save output stable videos", default="./"
    )
    parser.add_argument(
        "--OutNamePrefix", help="prefix name before the output video name", default=""
    )
    parser.add_argument(
        "--MaxLength",
        help="max number of frames can be dealt with one time",
        type=int,
        default=1200,
    )
    parser.add_argument(
        "--Repeat",
        help="max number of frames can be dealt with one time",
        type=int,
        default=50,
    )
    return parser.parse_args()


def generateStable(model, frames, outPath, outPrefix):
    # video_name = os.path.splitext(os.path.basename(base_path))[0]
    # output_folder = os.path.join(outPath, video_name)
    os.makedirs(outPath, exist_ok=True)

    # image_base_path = base_path
    # image_len = min(len([ele for ele in os.listdir(image_base_path) if ele[-4:] == '.jpg']), max_length)
    # image_len = min(len(frames), max_length)
    image_len = len(frames)
    # read input video
    frames = np.array(frames)
    images = []
    rgbimages = []
    index = 0
    for i in range(image_len):
        index += 1
        # print(i)
        # print(index)
        image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        image = image * (1.0 / 255.0)
        image = cv2.resize(image, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        images.append(image.reshape(1, 1, cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH))

        image = cv2.resize(frames[i], (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        rgbimages.append(np.expand_dims(np.transpose(image, (2, 0, 1)), 0))
        # # 根据索引生成文件名
        # image_filename = '{:06d}'.format(index)
        # # 构建完整的图像文件路径
        # image_path = os.path.join(image_base_path, image_filename + '.jpg')
        #
        # image = cv2.imread(image_path, 0)
        # image = image * (1. / 255.)
        # image = cv2.resize(image, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        # images.append(image.reshape(1, 1, cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH))
        #
        # # image = cv2.imread(os.path.join(image_base_path, '{}.jpg'.format(i)))
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        # rgbimages.append(np.expand_dims(np.transpose(image, (2, 0, 1)), 0))

    x = np.concatenate(images, 1).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)

    x_RGB = np.concatenate(rgbimages, 0).astype(np.float32)
    x_RGB = torch.from_numpy(x_RGB).unsqueeze(0)

    with torch.no_grad():
        # origin_motion, smoothPath = model.inference(x.cuda(), x_RGB.cuda(), repeat=args.Repeat)
        origin_motion, smoothPath = model.inference(x.cuda(), x_RGB.cuda(), repeat=50)

    origin_motion = origin_motion.cpu().numpy()
    smoothPath = smoothPath.cpu().numpy()
    origin_motion = np.transpose(origin_motion[0], (2, 3, 1, 0))
    smoothPath = np.transpose(smoothPath[0], (2, 3, 1, 0))

    x_paths = origin_motion[:, :, :, 0]
    y_paths = origin_motion[:, :, :, 1]
    sx_paths = smoothPath[:, :, :, 0]
    sy_paths = smoothPath[:, :, :, 1]

    frame_rate = 25
    frame_width = cfg.MODEL.WIDTH
    frame_height = cfg.MODEL.HEIGHT

    print("generate stabilized video...")
    print("outPath = " + outPath)
    # print("outPrefix = " + outPrefix)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(os.path.join(outPath, outPrefix + 'DUT_stable.mp4'), fourcc, frame_rate, (frame_width, frame_height))

    new_x_motion_meshes = sx_paths - x_paths
    new_y_motion_meshes = sy_paths - y_paths

    outImages = warpListImage(rgbimages, new_x_motion_meshes, new_y_motion_meshes)
    outImages = outImages.numpy().astype(np.uint8)
    outImages = [
        np.transpose(outImages[idx], (1, 2, 0)) for idx in range(outImages.shape[0])
    ]

    return outImages
    # for idx, frame in tqdm(enumerate(outImages)):
    #     # new_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
    #     cv2.imwrite(os.path.join(outPath, outPrefix + f"{idx}.jpg"), frame)

    # VERTICAL_BORDER = 60
    # HORIZONTAL_BORDER = 80

    # new_frame = frame[VERTICAL_BORDER:-VERTICAL_BORDER, HORIZONTAL_BORDER:-HORIZONTAL_BORDER]
    # new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

    # new_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
    # out.write(new_frame)

    # out.release()


def DUT_start(
    smootherPath,
    RFDetPath,
    PWCNetPath,
    MotionProPath,
    homo,
    frames,
    outPath,
    outPrefix,
    # maxlength,
):
    model = DUT(smootherPath, RFDetPath, PWCNetPath, MotionProPath, homo)
    model.cuda()
    model.eval()

    result = generateStable(model, frames, outPath, outPrefix)
    return result


if __name__ == "__main__":
    args = parse_args()
    print(args)

    smootherPath = args.SmootherPath
    RFDetPath = args.RFDetPath
    PWCNetPath = args.PWCNetPath
    MotionProPath = args.MotionProPath
    homo = not args.SingleHomo
    inPath = args.InputBasePath
    outPath = args.OutputBasePath
    outPrefix = args.OutNamePrefix
    maxlength = args.MaxLength

    model = DUT(
        SmootherPath=smootherPath,
        RFDetPath=RFDetPath,
        PWCNetPath=PWCNetPath,
        MotionProPath=MotionProPath,
        homo=homo,
    )
    model.cuda()
    model.eval()

    generateStable(model, inPath, outPath, outPrefix, maxlength)
