import os
import math
import time
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from config import DefaultConfig
from utils.det_frames.fcos import FCOSDetector
from utils.dataset.transforms import train_transforms
from utils.dataset.VOC import VOCDatasetMS


def train(opt, config=DefaultConfig):
    model = FCOSDetector(mode="training", config=config).cuda()
    model = torch.nn.DataParallel(model)
    model.train()
    print(model)

    train_dataset = VOCDatasetMS(config, root_dir=config.trainval_dir, transforms=train_transforms,
                                 split='trainval', use_difficult=False, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=opt.n_cpu,
                              worker_init_fn=np.random.seed(config.SEED))
    print("total_images : {}".format(len(train_dataset)))

    EPOCHS = config.EPOCHS
    STEPS_PER_EPOCH = math.ceil(len(train_dataset) / config.BATCH_SIZE)
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
    WARMPUP_STEPS = 501
    GLOBAL_STEPS = 1
    optimizer = config.optimizer_class(model.parameters(), **config.optimizer_params)

    folder = time.strftime("%Y%m%d__%H_%M", time.localtime())
    save_dir = os.path.join(config.exp_dir, folder, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    if config.save_best == "accuracy":
        best = 0.
    else:
        best = 999999.

    print("=" * 20, "start training", "=" * 20)
    for epoch in range(EPOCHS):
        epoch_start_time = time.strftime("%Y%m%d %H:%M", time.localtime())
        print("\r\nEPOCH: {}, start: {}".format(epoch+1, epoch_start_time))
        # cls, cnt, reg, total
        total_losses = [0., 0., 0., 0.]

        for epoch_step, data in enumerate(train_loader):
            batch_imgs, batch_boxes, batch_classes = data
            batch_imgs = batch_imgs.cuda()
            batch_boxes = batch_boxes.cuda()
            batch_classes = batch_classes.cuda()

            # warm up
            if GLOBAL_STEPS < WARMPUP_STEPS:
                lr = float(GLOBAL_STEPS / WARMPUP_STEPS * config.LR_INIT)
                for param in optimizer.param_groups:
                    param['lr'] = lr
            # lr decay
            elif GLOBAL_STEPS == int(TOTAL_STEPS*0.667):
                lr = config.LR_INIT * 0.1
                for param in optimizer.param_groups:
                    param['lr'] = lr
            elif GLOBAL_STEPS == int(TOTAL_STEPS*0.889):
                lr = config.LR_INIT * 0.01
                for param in optimizer.param_groups:
                    param['lr'] = lr

            optimizer.zero_grad()
            losses = model([batch_imgs, batch_boxes, batch_classes])
            loss = losses[-1]
            loss.mean().backward()
            optimizer.step()

            total_losses[-1] = 0.
            for i in range(len(total_losses)-1):
                total_losses[i] = (total_losses[i] + losses[i])
                total_losses[-1] += total_losses[i]

            if epoch_step % config.verbose_interval == 0:
                print('step: {:4d}/{}, lr: {:.7f}, '.format(epoch_step+1, STEPS_PER_EPOCH, lr) +
                      'cls: {:.5f}, cnt: {:.5f}, reg: {:.5f}, total: {:.5f}'.format(
                      *[l/(epoch_step+1) for l in total_losses]),
                      end='\r')
            GLOBAL_STEPS += 1

            # TODO validdation during training and save the best model

        torch.save(model.state_dict(), os.path.join(save_dir, "model_{}.pth".format(epoch + 1)))
        print("\r\n")


def fix_seed(config=DefaultConfig):
    os.environ['PYTHONHASHSEED'] = str(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='0', help="gpu during training")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_config = DefaultConfig
    fix_seed(train_config)

    train(args, train_config)













