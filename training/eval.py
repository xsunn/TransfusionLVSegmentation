import torch
import torch.nn.functional as F
from tqdm import tqdm

from loss import dice_coeff
import numpy as np

from loss import dice_coeff
from medpy.metric.binary import dc, hd, asd,jc

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader)  # the number of batch
    tot = []

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs,vel,true_masks = batch['image'],batch['vel'],batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            vel = vel.to(device=device, dtype=torch.float32)

            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs,vel)
                pred=mask_pred
                # pred = (mask_pred > 0.5).float()
                # print(pred.size())
                tot.append(dice_coeff(pred, true_masks).item())
            pbar.update()

    net.train()
    # print(len(tot))
    return np.mean(tot)

