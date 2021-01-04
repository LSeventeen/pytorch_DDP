import os
import random
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import torch

def dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)



def removeConnectedComponents(open, threshold, connectivity=8):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(open, connectivity=connectivity)
    results = np.zeros([labels.shape[0], labels.shape[1]], dtype='uint8')
    for h in range(labels.shape[0]):
        for w in range(labels.shape[1]):
            a = stats[labels[h, w], 4]
            if labels[h, w] != 0 and a > threshold:
                results[h, w] = 255

    return results


# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    img = None
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


def save_picture(img, file_name):
    for i in range(img.shape[0]):
        image = np.squeeze(img[i, :, :, :])
        image = Image.fromarray((image * 255).astype(np.uint8))  # the image is between 0-1
        image.save(file_name + str(i) + '.png')

class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break