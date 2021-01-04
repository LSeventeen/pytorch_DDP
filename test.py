import argparse
import os

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import models
from dataset import TestDataset
from trainer import Trainer


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def test(save_path, model_name):
    checkpoint = torch.load(os.path.join(save_path, model_name))
    config = checkpoint['config']
    # DATA LOADERS
    test_dataset = TestDataset(config["path"], config["data_set"]["name"])
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=4, sampler=test_sampler, pin_memory=True, num_workers=16)
    # MODEL
    model = get_instance(models, 'model', config)
    # TEST
    tester = Trainer(model=model, mode="test", config=config, checkpoint=checkpoint, test_loader=test_loader,
                     save_path=save_path, save_pic=True)
    tester.test()


def main():
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-m', '--model', default='model_weights.pth', type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:10044', rank=0, world_size=1)
    wandb.init(project="retina-segmentation", sync_tensorboard=True, name="test_")
    save_path = "saved/UNet_Multi/2012191322"
    model_name = "best_model.pth"
    test(save_path, model_name)


if __name__ == '__main__':
    main()
