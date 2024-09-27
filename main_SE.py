import PIL.Image
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import os
from partation_se import Partition
from argparse import ArgumentParser
from skimage.segmentation import relabel_sequential, find_boundaries
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
import csv
import torch
import torch.nn.functional as F
import numpy as np
import torch_scatter
from skimage.util import img_as_float
from scipy.io import loadmat

def get_args():
    DATASET = {
        "BSDS": ["BSDS500/images", "BSDS500/groundTruth"],
        "SBD": ["SBD/images", "SBD/groundTruth"],
        "PASCAL-S": ["PASCAL-S/images", "PASCAL-S/groundTruth"]
    }
    parser = ArgumentParser()

    parser.add_argument('--t', type=float,default=0.1, help=" ")
    parser.add_argument('--SE_t', type=float, default=2e-7, help=" ")
    parser.add_argument('--multi_scale', type=bool, default=True, help=" ")
    parser.add_argument('--target_size', type=int,default=[100,200,300,400,500,600,800,1000,1200,1500],
                        help="when multi_scale is True, the target_size must be a list like [100,200,300,400,500,600,800,1000,1200,1500]."
                             "Instead,when multi_scale is False, target_size must be an int value")

    cfg = parser.parse_args()
    cfg.dataset = DATASET
    return cfg

def draw_img(img,seg):
    imgH = img.shape[0]
    imgW = img.shape[1]
    img = img.cpu()
    seg = seg.cpu()
    torch.random.manual_seed(321)

    imgc = torch_scatter.scatter_mean(img.reshape(-1, 3), seg.reshape(-1), dim=0)

    imgs = imgc[seg].reshape(imgH, imgW, 3)

    imgcl = seg.reshape(1, 1, imgH, imgW).float()

    kernel1 = torch.Tensor([-1, 1, 0]).reshape(1, 1, 1, 3).cpu()
    kernel2 = torch.Tensor([-1, 1, 0]).reshape(1, 1, 3, 1).cpu()

    imgcl = torch.max(
        torch.abs(F.conv2d(imgcl, kernel1, padding=(0, 1))) + torch.abs(
            F.conv2d(imgcl, kernel2, padding=(1, 0))), 1)

    imgs[:, :, 0] += imgcl[0][0]
    imgs[:, :, 1] -= imgcl[0][0]
    imgs[:, :, 2] -= imgcl[0][0]

    imgs = imgs.clamp(max=1, min=0)
    imgs = np.array(imgs.cpu())
    return imgs

def start(data,gt,cfg):
    filenames = [filename for filename in os.listdir(data) if filename.endswith(".jpg")]

    for i, filename in enumerate(filenames):
        img_path = os.path.join(data, filename)
        img = PIL.Image.open(img_path)
        w, h = img.size
        img = np.array(img)[:, :, :3]
        img = torch.tensor(img).double() / 255

        segmentor = Partition(cfg.t, cfg.SE_t)
        seg = segmentor.fit(img, cfg.target_size,cfg.multi_scale)

        if cfg.multi_scale:
            for tgt_size,sub_seg in zip(cfg.target_size,seg):
                '''
                draw the picture
                '''
                imgs = draw_img(img, sub_seg)
                name, ext = os.path.splitext(os.path.basename(img_path))
                folder_path = os.path.join('./imgs', name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                imgPathOut = os.path.join(folder_path, name + "SE" + ' ' + str(tgt_size) + ext)
                plt.imsave(imgPathOut, imgs)

        else:
            imgs = draw_img(img, seg)
            name, ext = os.path.splitext(os.path.basename(img_path))
            folder_path = os.path.join('./imgs', name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            imgPathOut = os.path.join(folder_path, name + "SE" + ' ' + str(cfg.target_size) + ext)
            plt.imsave(imgPathOut, imgs)



if __name__ == '__main__':
    cfg = get_args()
    for d_name in ["BSDS","SBD","PASCAL-S"]:
        data,gt = cfg.dataset[d_name][0], cfg.dataset[d_name][1]
        start(data,gt,cfg)

