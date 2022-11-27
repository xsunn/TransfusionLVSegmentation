import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from model.loss import dice_coeff,DiceCoeff,DiceBCELoss
from eval import eval_net

from tensorboardX import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from model.SegModel import seg

#training and validation data path
valFilePath=".../valFolder1.txt"
trainFilePath=".../trainFolder1.txt"

# model weight path
dir_checkpoint=".../modelWeight//"

def train_net(net,
              device,
              epochs=2,
              batch_size=200,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    train = BasicDataset(trainFilePath)
    val = BasicDataset(valFilePath)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    n_train=len(train)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train)}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    init_score=0.
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        evalDice=[]
        with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                vel=batch['vel']
                imgs = imgs.to(device=device, dtype=torch.float32)
                vel = vel.to(device=device, dtype=torch.float32)

                mask_type = torch.float32
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs,vel)
                loss = DiceBCELoss(masks_pred, true_masks)
                epoch_loss += loss.item()

                dice= dice_coeff(masks_pred, true_masks).item()
                evalDice.append(dice)
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item(),'Fusdice (batch)': dice})

                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        val_score = eval_net(net, val_loader, device)

        print(f'epoch{epoch } training dice is ' ,np.mean(evalDice))
        print(f'epoch{epoch } validation dice is ' ,val_score)
        print(f'epoch{epoch } best dice is ' ,init_score)

        if val_score>= init_score:
            init_score=val_score
            save_cp=True
        else:
            save_cp=False
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'modelWeight.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            # print("val dice is",init_score)
    writer.close()
#
#
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=3000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=12,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.000001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(torch.cuda.device_count() )
    logging.info(f'Using device {device}')


    net=seg()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)


    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)