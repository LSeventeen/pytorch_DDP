import json
import math
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from loguru import logger
from matplotlib import pyplot as plt
from torch.utils import tensorboard
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.helpers import dir_exists, group_images, visualize
from utils.metrics import (AverageMeter, eval_metrics, get_metrics,
                           get_metrics_full)


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Trainer:
    def __init__(self, mode, model, rank, resume=None, config=None, loss=None,

                 train_loader=None,
                 val_loader=None,
                 checkpoint=None,
                 test_loader=None,

                 save_path=None, show=False, save_pic=False):
        self.rank = rank
        self.config = config

        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.group = 4
        self.save_pic = save_pic
        self.gt_num = config["loss"]["gt_num"]
        self.model = model

        if self.rank == 0: wandb.watch(self.model)

        cudnn.benchmark = True
        # train and val
        if mode == "train":
            self.start_epoch = 1
            self.show = show
            self.loss = loss
            # OPTIMIZER
            self.optimizer = getattr(torch.optim, config['optimizer']['type'])(self.model.parameters(),
                                                                               **config['optimizer']['args'])
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(
                self.optimizer, **config['lr_scheduler']['args'])

            # summary(model, input_size=(
            #     1, self.config["data_set"]["patch_size"], self.config["data_set"]["patch_size"]))
            # CONFIGS

            cfg_trainer = self.config['trainer']
            self.epochs = cfg_trainer['epochs']
            if self.rank == 0:
                self.save_period = cfg_trainer['save_period']
                # MONITORING
                self.improved = True
                self.not_improved_count = 0
                self.monitor = cfg_trainer.get('monitor', 'off')
                if self.monitor == 'off':
                    self.mnt_mode = 'off'
                    self.mnt_best = 0
                else:
                    self.mnt_mode, self.mnt_metric, self.gap = self.monitor.split()

                    assert self.mnt_mode in ['min', 'max']
                    self.mnt_best = -math.inf if self.mnt_mode == 'max' else math.inf
                    self.early_stopping = cfg_trainer.get('early_stop', math.inf)

                # CHECKPOINTS & TENSOBOARD

                start_time = datetime.now().strftime('%y%m%d%H%M')
                self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], self.config['model']['type'], start_time)
                self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
                dir_exists(self.checkpoint_dir)
                config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
                self.train_logger_save_path = os.path.join(self.checkpoint_dir, 'runtime.log')
                logger.add(self.train_logger_save_path)
                logger.info(self.checkpoint_dir)
                with open(config_save_path, 'w') as handle:
                    json.dump(self.config, handle, indent=4, sort_keys=True)
                self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
                self.log_step = config['trainer'].get('log_per_iter', self.train_loader.batch_size)
                if resume: self._resume_checkpoint(resume)

        # test
        if mode == "test":
            self.model.load_state_dict(checkpoint['state_dict'])
            self.checkpoint_dir = save_path

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            # RUN TRAIN (AND VAL)
            self._train_epoch(epoch)
            if self.rank == 0 and self.val_loader is not None and epoch % self.config['trainer']['val_per_epochs'] == 0:
                results = self._valid_epoch(epoch)
                # LOGGING INFO

                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'         {str(k):15s}: {v}')
                # CHECKING IF THIS IS THE BEST MODEL (ONLY FOR VAL)
                if self.mnt_mode != 'off' and epoch >= 10:
                    try:
                        if self.mnt_mode == 'min':
                            self.improved = (results[self.mnt_metric] <= self.mnt_best - float(self.gap))
                        else:
                            self.improved = (results[self.mnt_metric] >= self.mnt_best + float(self.gap))
                    except KeyError:
                        logger.warning(
                                f'The metrics being tracked ({self.mnt_metric}) has not been calculated. Training stops.')
                        break

                    if self.improved:
                        self.mnt_best = results[self.mnt_metric]
                        self.not_improved_count = 0

                    else:
                        self.not_improved_count += 1

                    if self.not_improved_count >= self.early_stopping:
                        logger.info(f'\nPerformance didn\'t improve for {self.early_stopping} epochs')
                        logger.warning('Training Stoped')
                        break
                    # SAVE CHECKPOINT

            # if epoch % self.save_period == 0:
            #     if self.rank == 0:
            #         checkpoint_path = self._save_checkpoint(epoch, save_best=self.improved)
            #     else:
            #         dist.barrier()
            #         map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            #         self.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location)['state_dict'])
        if self.test_loader is not None:
            self.test(log="tensorboard")

    def _train_epoch(self, epoch):

        self.model.train()
        if self.rank == 0:
            wrt_mode = 'train'

            y_true = []
            y_score = []
            y_score_b = []

            tic = time.time()
            self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=160)
        for batch_idx, (img, gt, Sgt, Lgt, mask) in enumerate(tbar):
            if self.rank == 0: self.data_time.update(time.time() - tic)
            img = img.to(self.rank, non_blocking=True)
            gt = gt.to(self.rank, non_blocking=True)
            mask = mask.to(self.rank, non_blocking=True)
            Sgt = Sgt.to(self.rank, non_blocking=True)
            Lgt = Lgt.to(self.rank, non_blocking=True)
            # LOSS & OPTIMIZE

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                if self.gt_num == 1:
                    predict = self.model(img)
                    loss = self.loss(predict, gt)
                elif self.gt_num == 2:
                    s, predict = self.model(img)
                    loss = self.loss(predict, gt, s, Sgt)
                else:
                    l, s, predict = self.model(img)
                    loss = self.loss(predict, gt, s, Sgt, l, Lgt)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.rank == 0:
                self.total_loss.update(loss.item())
                # measure elapsed time
                self.batch_time.update(time.time() - tic)
                tic = time.time()
                # LOGGING & TENSORBOARD
                if batch_idx % self.log_step == 0:
                    wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx

                predict = torch.sigmoid(predict).cpu().detach().numpy().ravel()
                predict_b = np.where(predict >= 0.5, 1, 0)
                # predict_b = torch.where(predict >= 0.5, torch.full_like(predict, 1), torch.full_like(predict, 0))
                mask = mask.cpu().detach().numpy().ravel()
                y_true = gt.cpu().detach().numpy().ravel()[mask == 1]
                y_score = predict[mask == 1]
                y_score_b = predict_b[mask == 1]

                # FOR EVAL and INFO
                if self.rank == 0:
                    self._update_seg_metrics(*eval_metrics(y_true, y_score_b))
                    metrics = get_metrics(self.tn, self.fp, self.fn, self.tp)
                    tbar.set_description(
                        'TRAIN ({}) | Loss: {:.4f} | Acc {:.4f} Pre {:.4f} Sen {:.4f} Spe {:.4f} f1 {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                            epoch, self.total_loss.average, *metrics.values(), self.batch_time.average,
                            self.data_time.average))

            # METRICS TO TENSORBOARD
        if self.rank == 0:
            metrics = get_metrics_full(self.tn, self.fp, self.fn, self.tp, y_true, y_score, y_score_b)
            self.writer.add_scalar(f'{wrt_mode}/loss', self.total_loss.average, epoch)
            for k, v in list(metrics.items())[:-1]:
                self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
            # self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)

        self.lr_scheduler.step()

    def _valid_epoch(self, epoch):
        if self.rank == 0:
            logger.info('\n###### EVALUATION ######')
            wrt_mode = 'val'
            self._reset_metrics()
            val_img = []
            y_true = []
            y_score = []
            y_score_b = []
        self.model.eval()
        tbar = tqdm(self.val_loader, ncols=160)
        with torch.no_grad():

            for batch_idx, (img, gt, Sgt, Lgt, mask) in enumerate(tbar):
                img = img.to(self.rank, non_blocking=True)
                gt = gt.to(self.rank, non_blocking=True)
                mask = mask.to(self.rank, non_blocking=True)
                Sgt = Sgt.to(self.rank, non_blocking=True)
                Lgt = Lgt.to(self.rank, non_blocking=True)
                # LOSS

                with torch.cuda.amp.autocast(enabled=True):
                    if self.gt_num == 1:
                        predict = self.model(img)
                        loss = self.loss(predict, gt)
                    elif self.gt_num == 2:
                        s, predict = self.model(img)
                        loss = self.loss(predict, gt, s, Sgt)
                    else:
                        l, s, predict = self.model(img)
                        loss = self.loss(predict, gt, s, Sgt, l, Lgt)
                if self.rank == 0:
                    self.total_loss.update(loss.item())
                    predict = torch.sigmoid(predict).cpu().detach().numpy()
                    predict_b = np.where(predict >= 0.5, 1, 0)
                    mask = mask.cpu().detach().numpy().ravel()
                    y_true = gt.cpu().detach().numpy().ravel()[mask == 1]
                    y_score = predict.ravel()[mask == 1]
                    y_score_b = predict_b.ravel()[mask == 1]
                    # FOR EVAL and INFO
                    self._update_seg_metrics(*eval_metrics(y_true, y_score_b))
                    metrics = get_metrics(self.tn, self.fp, self.fn, self.tp)
                    tbar.set_description(
                        'EVAL ({}) | Loss: {:.4f} | Acc {:.4f} Pre {:.4f} Sen {:.4f} Spe {:.4f} f1 {:.4f} IOU {:.4f} |'.format(
                            epoch, self.total_loss.average, *metrics.values()))

                    # LIST OF IMAGE TO VIZ (15 images)

                    if batch_idx < 10:
                        val_img.extend([img[0].data.cpu(), gt[0].data.cpu(), torch.tensor(predict_b[0])])
            if self.rank == 0:
                val_img = torch.stack(val_img, 0)
                val_img = make_grid(val_img, nrow=3, padding=2)
                if self.show is True:
                    plt.figure(figsize=(12, 36))
                    plt.imshow(transforms.ToPILImage()(val_img.squeeze(0)).convert('L'), cmap='gray')
                    plt.show()

                # LOGGING & TENSORBOARD
                wrt_step = epoch
                metrics = get_metrics_full(self.tn, self.fp, self.fn, self.tp, y_true, y_score, y_score_b)
                self.writer.add_image(f'{wrt_mode}/inputs_targets_predictions', val_img, wrt_step)
                self.writer.add_scalar(f'{wrt_mode}/loss', self.total_loss.average, wrt_step)
                for k, v in list(metrics.items())[:-1]:
                    self.writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)
                log = {
                    'val_loss': self.total_loss.average,
                    **metrics
                }
        return log

    def test(self, epoch=1, log="wandb"):
        # self._test_reset_metrics()
        logger.info('###### TEST EVALUATION ######')
        wrt_mode = 'test'
        self.model.eval()
        tbar = tqdm(self.test_loader, ncols=50)
        imgs = []
        predicts = []
        gts = []
        masks = []
        tic1 = time.time()

        with torch.no_grad():
            for batch_idx, (img, gt, mask), in enumerate(tbar):
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    if self.gt_num == 1:
                        predict = self.model(img)
                    elif self.gt_num == 2:
                        _, predict = self.model(img)
                    else:
                        _, _, predict = self.model(img)
                img = img.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
                gt = gt.cpu().detach().numpy()
                predict = torch.sigmoid(predict).cpu().detach().numpy()

                imgs.extend(img)
                gts.extend(gt)
                predicts.extend(predict)
                masks.extend(mask)

            imgs = np.asarray(imgs)
            gts = np.asarray(gts)
            predicts = np.asarray(predicts)
            masks = np.asarray(masks)
        tic2 = time.time()
        test_time = tic2 - tic1
        logger.info(f'test time:  {test_time}')
        predicts_b = np.where(predicts >= 0.5, 1, 0)
        if self.save_pic is True:
            assert (predicts.shape[0] % self.group == 0)

            for i in range(int(predicts.shape[0] / self.group)):
                # orig_rgb_stripe = group_images(
                #     self.test_imgs_original[i * self.group:(i * self.group) + self.group, :, :, :], self.group)
                orig_stripe = group_images(imgs[i * self.group:(i * self.group) + self.group, :, :, :], self.group)
                gt_stripe = group_images(gts[i * self.group:(i * self.group) + self.group, :, :, :], self.group)
                pred_stripe = group_images(predicts_b[i * self.group:(i * self.group) + self.group, :, :, :],
                                           self.group)
                total_img = np.concatenate(
                    (np.tile(orig_stripe, 3), np.tile(gt_stripe, 3), np.tile(pred_stripe, 3)), axis=0)
                if self.save_pic is True:
                    visualize(total_img, self.checkpoint_dir + "/RGB_Original_GroundTruth_Prediction" + str(i))

        # LOGGING & TENSORBOARD
        wrt_step = epoch
        metrics = get_metrics_full(*eval_metrics(gts[masks == 1], predicts_b[masks == 1]), gts[masks == 1],
                                   predicts[masks == 1], predicts_b[masks == 1])
        # self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', total_img, self.wrt_step)
        tic3 = time.time()
        metrics_time = tic3 - tic1
        logger.info(f'metrics time:  {metrics_time}')
        # self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)

        for k, v in list(metrics.items())[:-1]:
            if log == "wandb":
                wandb.log({f'{wrt_mode}/{k}': v})
            else:
                self.writer.add_scalar(f'{wrt_mode}/{k}', v, wrt_step)
        for k, v in metrics.items():
            logger.info(f'         {str(k):15s}: {v}')

    def _save_checkpoint(self, epoch, save_best=True):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)

        if save_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pth')
            torch.save(state, filename)
            logger.info("Saving current best: best_model.pth")
        return filename

    def _resume_checkpoint(self, resume_path):
        logger.info(f'Loading checkpoint : {resume_path}')
        checkpoint = torch.load(resume_path)

        # Load last run info, the model params, the optimizer and the loggers
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.not_improved_count = 0

        if checkpoint['config']['mode']['type'] != self.config['mode']['type']:
            logger.warning({'Warning! Current model is not the same as the one in the checkpoint'})
        self.model.load_state_dict(checkpoint['state_dict'])

        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            logger.warning({'Warning! Current optimizer is not the same as the one in the checkpoint'})
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # if self.lr_scheduler:
        #     self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.train_logger = checkpoint['logger']
        logger.info(f'Checkpoint <{resume_path}> (epoch {self.start_epoch}) was loaded')

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.tp, self.tn = 0, 0
        self.fp, self.fn = 0, 0

    def _update_seg_metrics(self, tn, fp, fn, tp):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn
