import random

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, ToPILImage, CenterCrop, ColorJitter

from data_process import read_pickle
from utils.myTransforms import MyRandomRotation, MyRandomHorizontalFlip, MyRandomVerticalFlip


class TrainDataset(Dataset):
    def __init__(self, path, name, patch_size, stride, split, mode=None):
        self.mode = mode
        if name == "DRIVE":
            img, gt, mask, Sgt, Lgt = read_pickle(path + "/DRIVE/training")
        elif name == "CHASEDB1":
            img, gt, mask, Sgt, Lgt = read_pickle(path + "/CHASEDB1")
            img = img[0:20]
            gt = gt[0:20]
            mask = mask[0:20]
            Sgt = Sgt[0:20]
            Lgt = Lgt[0:20]
        elif name == "STARE":
            img, gt, mask, Sgt, Lgt = read_pickle(path + "STARE")
        self.img_p = self._get_patch(img, patch_size, stride)
        self.gt_p = self._get_patch(gt, patch_size, stride)
        self.mask_p = self._get_patch(mask, patch_size, stride)
        self.Sgt_p = self._get_patch(Sgt, patch_size, stride)
        self.Lgt_p = self._get_patch(Lgt, patch_size, stride)
        if self.mode == "train":
            self.img = self.img_p[0:int(len(self.img_p) * split)]
            self.gt = self.gt_p[0:int(len(self.gt_p) * split)]
            self.Sgt = self.Sgt_p[0:int(len(self.Sgt_p) * split)]
            self.Lgt = self.Lgt_p[0:int(len(self.Lgt_p) * split)]
            self.mask = self.mask_p[0:int(len(self.mask_p) * split)]
        elif self.mode == "val":
            self.img = self.img_p[int(len(self.img_p) * split):]
            self.gt = self.gt_p[int(len(self.gt_p) * split):]
            self.Sgt = self.Sgt_p[int(len(self.Sgt_p) * split):]
            self.Lgt = self.Lgt_p[int(len(self.Lgt_p) * split):]
            self.mask = self.mask_p[int(len(self.mask_p) * split):]

        self.img_transforms = Compose([

            ToPILImage(),
            MyRandomHorizontalFlip(p=0.5),
            MyRandomVerticalFlip(p=0.5),
            MyRandomRotation([0, 90, 180, 270]),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # RandomElastic(alpha=0.2, sigma=0.1),
            # RandomAffineCV2(alpha=0.005),
            ToTensor(),

        ])

        self.gt_transforms = Compose([
            ToPILImage(),
            MyRandomHorizontalFlip(p=0.5),
            MyRandomVerticalFlip(p=0.5),
            MyRandomRotation([0, 90, 180, 270]),
            # RandomElastic(alpha=0.2, sigma=0.1),
            # RandomAffineCV2(alpha=0.005),
            ToTensor(),
        ])

    def _get_patch(self, imgs_list, patch_size, stride):
        image_list = []
        for i in imgs_list:
            image = i.unfold(1, patch_size, stride).unfold(2, patch_size, stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
            for sub_mask in image:
                image_list.append(sub_mask)
        return image_list

    def __getitem__(self, idx):
        # seed so image and target have the same random tranform
        seed = random.randint(0, 2 ** 32)

        img = self.img[idx]
        mask = self.mask[idx]
        gt = self.gt[idx]
        Sgt = self.Sgt[idx]
        Lgt = self.Lgt[idx]
        if self.mode != "val":
            random.seed(seed)
            img = self.img_transforms(img)
            random.seed(seed)
            mask = self.gt_transforms(mask)
            random.seed(seed)
            gt = self.gt_transforms(gt)
            random.seed(seed)
            Sgt = self.gt_transforms(Sgt)
            random.seed(seed)
            Lgt = self.gt_transforms(Lgt)

            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 3, 1)
            # ax1.imshow(torch.squeeze(img), cmap="gray")
            # ax2 = fig.add_subplot(1, 3, 2)
            # ax2.imshow(torch.squeeze(mask), cmap="gray")
            # ax3 = fig.add_subplot(1, 3, 3)
            # ax3.imshow(torch.squeeze(target), cmap="gray")
            # plt.show()

        return img, gt, Sgt, Lgt, mask,

    def __len__(self):
        return len(self.img)


class TestDataset(Dataset):
    def __init__(self, path, name):
        self.name = name
        if name == "DRIVE":
            self.img, self.gt, self.mask, _, _ = read_pickle(path + "/DRIVE/test")
        elif name == "CHASEDB1":
            img, gt, mask, _, _ = read_pickle(path + "/CHASEDB1")
            self.img = img[20:28]
            self.gt = gt[20:28]
            self.mask = mask[20:28]

        # size = np.median(self.img[0].shape) - 1 if np.median(self.img[0].shape) % 2 == 0 else np.median(
        #     self.image_list[0].shape)

        self.transforms = Compose([
            ToPILImage(),
            CenterCrop(512),
            ToTensor(),
        ])

    def __getitem__(self, idx):
        img = self.img[idx]
        gt = self.gt[idx]
        mask = self.mask[idx]
        img = self.transforms(img)
        gt = self.transforms(gt)
        mask = self.transforms(mask)
        return img, gt, mask

    def __len__(self):
        return len(self.img)


if __name__ == '__main__':
    path = "/home/lwt/data/DRIVE"
    # image_list = CHASE_read(path=path, type="image")
    data = TestDataset(path, "DRIVE")
