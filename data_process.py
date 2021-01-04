import argparse
import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
from ruamel.yaml import YAML
from scipy import ndimage, stats
from skimage import color, measure
from torchvision.transforms import ToTensor, Normalize, Grayscale

from utils.helpers import removeConnectedComponents


def data_process(path, name, mode=None, se_size=3, remove_size=200):
    if name == "DRIVE":
        img_path = os.path.join(path, name, mode, "images")
        gt_path = os.path.join(path, name, mode, "1st_manual")
        mask_path = os.path.join(path, name, mode, "mask")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        path=os.path.join(path, name)
        file_list = list(sorted(os.listdir(path)))
    elif name == "STARE":
        img_path = os.path.join(path, name, "stare-images")
        gt_path = os.path.join(path, name, "labels-ah")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    mask_list = []

    for file in file_list:
        if name == "DRIVE":
            gt_name = "_manual1.gif"
            if mode == "training":
                mask_name = "_training_mask.gif"
            else:
                mask_name = "_test_mask.gif"
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:2] + gt_name))
            mask = Image.open(os.path.join(mask_path, file[0:2] + mask_name))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
            mask_list.append(ToTensor()(mask))
        elif name == "CHASEDB1":
            if len(file) == 13:
                img = Image.open(os.path.join(path, file))
                mask = get_fov_mask(img, 0.01)
                gt = Image.open(os.path.join(path, file[0:9] + '_1stHO.png'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
                mask_list.append(ToTensor()(mask))
        elif name == "STARE":
            if len(file) == 10:
                img = Image.open(os.path.join(img_path, file))
                mask = get_fov_mask(img, 0.19)
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
                mask_list.append(ToTensor()(mask))

        # plt.figure(figsize=(12, 36))
        # plt.subplot(311)
        # plt.imshow(img, cmap='gray')
        # plt.subplot(312)
        # plt.imshow(gt, cmap='gray')
        # plt.subplot(313)
        # plt.imshow(mask, cmap='gray')
        # plt.show()

    Sgt_list, Lgt_list = small_gt(gt_list, se_size, remove_size)
    mean, std = getMeanStd(img_list, mask_list)
    img_list = normalization(img_list, mean, std)
    return img_list, gt_list, mask_list, Sgt_list, Lgt_list


def save_pickle(path, img_list, gt_list, mask_list, Sgt_list, Lgt_list):
    with open(file=path + "/img.pkl", mode='wb') as file:
        pickle.dump(img_list, file)
    with open(file=path + "/gt.pkl", mode='wb') as file:
        pickle.dump(gt_list, file)
    with open(file=path + "/mask.pkl", mode='wb') as file:
        pickle.dump(mask_list, file)
    with open(file=path + "/Sgt.pkl", mode='wb') as file:
        pickle.dump(Sgt_list, file)
    with open(file=path + "/Lgt.pkl", mode='wb') as file:
        pickle.dump(Lgt_list, file)


def read_pickle(path):
    with open(file=path + "/img.pkl", mode='rb') as file:
        img = pickle.load(file)
    with open(file=path + "/gt.pkl", mode='rb') as file:
        gt = pickle.load(file)
    with open(file=path + "/mask.pkl", mode='rb') as file:
        mask = pickle.load(file)
    with open(file=path + "/Sgt.pkl", mode='rb') as file:
        Sgt = pickle.load(file)
    with open(file=path + "/Lgt.pkl", mode='rb') as file:
        Lgt = pickle.load(file)
    return img, gt, mask, Sgt, Lgt


def small_gt(gt_list, se_size, remove_size):
    gt_small_list = []
    gt_large_list = []
    for i, gt in enumerate(gt_list):
        gt = np.squeeze(np.asarray(gt, dtype=np.uint8)) * 255
        ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
        open = cv2.morphologyEx(gt, cv2.MORPH_OPEN, ELLIPSE)
        gt_large = removeConnectedComponents(open, remove_size)
        sub = gt - gt_large
        # gt_small = removeConnectedComponents(sub, remove_size2)
        gt_large = torch.from_numpy(gt_large / 255).float()
        gt_small = torch.from_numpy(sub / 255).float()

        gt_large_list.append(gt_large.unsqueeze(dim=0))
        gt_small_list.append(gt_small.unsqueeze(dim=0))
    return gt_small_list, gt_large_list


def getMeanStd(imgs_list, mask_list=None):
    # global image_mask
    # for i in range(len(imgs_list)):
    #     image_select = torch.masked_select(imgs_list[i], mask_list[i].bool())
    #     if i == 0:
    #         image_mask = image_select
    #     else:
    #         image_mask = torch.cat((image_mask, image_select))
    # mean = torch.mean(image_mask)
    # std = torch.std(image_mask)
    imgs_list = torch.stack(imgs_list, dim=0)

    mean = torch.mean(imgs_list)
    std = torch.std(imgs_list)
    return mean, std


def normalization(imgs_list, mean, std):
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
        # plt.figure(figsize=(12, 36))
        # plt.subplot(211)
        # plt.imshow(torch.squeeze(i), cmap='gray')
        # plt.subplot(212)
        # plt.imshow(torch.squeeze(n), cmap='gray')
        # plt.show()
    return normal_list


def get_fov_mask(image_rgb, threshold=0.01):
    '''
    Automatically calculate the FOV mask (see Orlando et al., SIPAIM 2016 for further details) Convolutional neural network transfer for automated glaucoma identification
    '''
    # format: [H, W, #channels]
    image_lab = color.rgb2lab(image_rgb)
    # normalize the luminosity plane
    image_lab[:, :, 0] /= 100.0
    # threshold the plane at the given threshold
    mask = image_lab[:, :, 0] >= threshold
    # fill holes in the resulting mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.filters.median_filter(mask, size=(5, 5))
    # get connected components
    connected_components = measure.label(mask).astype(float)
    # replace background found in [0][0] to nan so mode skips it
    connected_components[connected_components == mask[0][0]] = np.nan
    # get largest connected component (== mode of the image)
    largest_component_label = stats.mode(connected_components, axis=None, nan_policy='omit')[0]
    # use the modal value of the labels as the final mask
    mask = connected_components == largest_component_label
    return mask.astype(float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open('config.yaml', encoding='utf-8') as file:
        config = yaml.load(file)  # 为列表类型
    save_pickle(config["path"] + "/DRIVE/training",
                *data_process(config["path"], name="DRIVE", mode="training", **config["data_process"]))
    save_pickle(config["path"] + "/DRIVE/test",
                *data_process(config["path"], name="DRIVE", mode="test", **config["data_process"]))
    save_pickle(config["path"] + "/CHASEDB1",
                *data_process(config["path"], name="CHASEDB1", **config["data_process"]))
    save_pickle(config["path"] + "/STARE",
                *data_process(config["path"], name="STARE", **config["data_process"]))
    # img, gt, mask, Sgt = read_pickle(config["path"] + "/DRIVE/training")
    # img2, gt2, mask2, Sgt2 = read_pickle(config["path"] + "/DRIVE/test")
    # img3, gt3, mask3, Sgt3 = read_pickle(config["path"] + "/CHASEDB1")
    # img4, gt4, mask4, Sgt4 = read_pickle(config["path"] + "/STARE")
