import argparse
import os
import sys

import torch.distributed as dist
import wandb
from loguru import logger
from ruamel.yaml import YAML
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import models
from dataset import TestDataset, TrainDataset
from trainer import Trainer
from utils import losses


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    if sys.platform == 'win32':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        # Example init_method="file:///f:/libtmp/some_file"
        init_method = "file:///{your local file path}"

        # initialize the process group
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def Train(rank, world_size, config, resume):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:10006', rank=rank, world_size=world_size)

    batch_size = int(config['data_loader']["batch_size"] / world_size)
    train_dataset = TrainDataset(mode="train", path=config["path"], **config["data_set"])
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8,
                              pin_memory=True)

    val_dataset = TrainDataset(mode="val", path=config["path"], **config["data_set"])
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8,
                            pin_memory=True)

    test_dataset = TestDataset(path=config["path"], name=config["data_set"]["name"])
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers=8,
                             pin_memory=True, )

    # MODEL
    # model = EfficientNet.from_name('efficientnet-b0', n_classes=1, pretrained=False).cuda()
    model = get_instance(models, 'model', config).to(rank)
    model = DDP(model, device_ids=[rank])
    if rank == 0:
        wandb.init(project="retina-segmentation", config=config, sync_tensorboard=True, name=" UNet_oetd celoss3")
        logger.info('The patch number of train is %d' % len(train_dataset))
        logger.info(f'\n{model}\n')
    # LOSS
    loss = get_instance(losses, 'loss', config)

    # TRAINING
    trainer = Trainer(
        model=model,
        rank=rank,
        mode="train",
        loss=loss,
        config=config,
        resume=resume,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    trainer.train()
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    yaml = YAML(typ='safe')
    with open('config.yaml', encoding='utf-8') as file:
        config = yaml.load(file)  # 为列表类型
    if args.resume:
        with open('config.yaml', encoding='utf-8') as file:
            config = yaml.load(args.resume)['config']
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    # if args.device:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001', rank=0, world_size=1)

    # wandb.init(project="retina-segmentation", config=config, sync_tensorboard=True, name=" UNet_oetd celoss3")

    n_gpus = torch.cuda.device_count()

    Train(args.local_rank, n_gpus, config, args.resume)
    # mp.spawn(Train, args=(n_gpus, config, args.resume), nprocs=n_gpus, join=True)
