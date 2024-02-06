import os
import time
import datetime

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import matplotlib
matplotlib.use('tkagg')
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from eval.eval_HDNet import eval_net
from utils.dataset import BuildingDataset
from torch.utils.data import DataLoader
from utils.sync_batchnorm.batchnorm import convert_model
from model.HDNet import HighResolutionDecoupledNet
from PIL import Image
import numpy as np
import random

torch.set_num_threads(16)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch HDNet training")
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=150, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--data-path", default="/media/bimeiqiao/sda11/liyuxuan/data/Massachusetts/")
    parser.add_argument("--numworkers", default=0, type=int)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--base-channel", default=48, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--read-name", default='')
    parser.add_argument("--save-name", default='HDNet_Mass_test')
    parser.add_argument("--DataSet", default='Mass')
    args = parser.parse_args()

    return args

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) / (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


def criterion(inputs, target, loss_weight=torch.tensor(1), dice: bool = True, size=512):
    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    if size == 512:
        loss = bcecriterion(inputs.squeeze(), target.squeeze().float())
    else:
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        target = F.interpolate(target, mode='bilinear', size=(size, size))
        loss = bcecriterion(inputs.squeeze(), target.squeeze().float())
    if dice is True:
        loss += dice_loss_func(torch.sigmoid(inputs.squeeze()), target.squeeze().float())
    return loss


def train_net(read_name,
              save_name,
              DataSet,
              net,
              device,
              data_path,
              epochs=5,
              batch_size=1,
              lr=0.001,
              num_workers=24,
              save_weights=True):
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    traindataset = BuildingDataset(dataset_dir=data_path, training=True, txt_name="final_train.txt", data_name=args.DataSet)
    valdataset = BuildingDataset(dataset_dir=data_path, training=False, txt_name="val.txt", data_name=args.DataSet)
    train_loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)
    val_loader = DataLoader(valdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(traindataset)}
        Validation size: {len(valdataset)}
        Saveweights:     {save_weights}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.module.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.7)

    print('Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])

    if os.path.exists(os.path.join('save_weights', read_name + '.pth')):
        best_val_score = eval_net(net, val_loader, device, savename=DataSet + '_' + read_name)  #
        print('Best iou:', best_val_score)
        no_optim = 0
    else:
        print('Training new model....')
        best_val_score = -1

    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=len(traindataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for num, batch in enumerate(train_loader):
                imgs = batch['image']
                true_labels = batch['label'] > 0
                dis_masks = batch['distance_map']
                imgs = imgs.to(device=device, dtype=torch.float32)
                label_type = torch.float32
                true_labels = true_labels.to(device=device, dtype=label_type)
                dis_masks = dis_masks.to(device=device).float()
                edge_masks = ((dis_masks < 3) & (dis_masks > 0)).to(device=device).float()
                x_seg, x_bd, seg1, seg2, seg3, seg4, seg5, seg6, bd1, bd2, bd3, bd4, bd5, bd6 = net(imgs)

                # Mass: 3 / 9 WHU: 7 / 21 Inria: 10 / 30
                loss = criterion(x_seg, true_labels, dice=True) + \
                       0.3 * criterion(seg1, true_labels, dice=True, size=256) + \
                       0.3 * criterion(seg2, true_labels, dice=True, size=256) + \
                       0.5 * criterion(seg3, true_labels, dice=True, size=256) + \
                       0.5 * criterion(seg4, true_labels, dice=True, size=256) + \
                       0.5 * criterion(seg5, true_labels, dice=True, size=256) + \
                       0.5 * criterion(seg6, true_labels, dice=True) + \
                       criterion(x_bd, edge_masks, loss_weight=torch.tensor(9), dice=True) + \
                       0.3 * criterion(bd1, edge_masks, loss_weight=torch.tensor(3), dice=True, size=256) + \
                       0.3 * criterion(bd2, edge_masks, loss_weight=torch.tensor(3), dice=True, size=256) + \
                       0.5 * criterion(bd3, edge_masks, loss_weight=torch.tensor(3), dice=True, size=256) + \
                       0.5 * criterion(bd4, edge_masks, loss_weight=torch.tensor(3), dice=True, size=256) + \
                       0.5 * criterion(bd5, edge_masks, loss_weight=torch.tensor(3), dice=True, size=256) + \
                       0.5 * criterion(bd6, edge_masks, loss_weight=torch.tensor(9), dice=True)

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                # with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
            val_score = eval_net(net, val_loader, device)

            with open(results_file, "a") as f:
                info = f"[epoch: {epoch}]\n" \
                       f"batch_loss: {loss.item():.4f}\n" \
                       f"val_IoU: {val_score:.6f}\n"
                f.write(info + "\n\n")

            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(net.module.state_dict(),
                           dir_checkpoint + save_name + '_best.pth')
                logging.info(f'Checkpoint {save_name} saved !')
                no_optim = 0
            else:
                no_optim = no_optim + 1
            # torch.save(net.module.state_dict(), dir_checkpoint + save_name + "_model_{}.pth".format(epoch))
            # logging.info(f'Checkpoint {save_name} saved !')

            if no_optim > 3:
                net.module.load_state_dict(torch.load(dir_checkpoint + save_name + '_best.pth'))
                scheduler.step()
                print('Scheduler step!')
                print('Learning rate: ', optimizer.state_dict()['param_groups'][0]['lr'])
                no_optim = 0

            if optimizer.state_dict()['param_groups'][0]['lr'] < 1e-7:
                break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def main(args):
    read_name = args.read_name
    save_name = args.save_name
    Dataset = args.DataSet
    assert Dataset in ['WHU', 'Inria', 'Mass']
    print(save_name)
    net = HighResolutionDecoupledNet(base_channel=args.base_channel, num_classes=args.num_classes)
    print('HDNet parameters: %d' % sum(p.numel() for p in net.parameters()))
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if read_name != '':
        net_state_dict = net.state_dict()
        state_dict = torch.load(dir_checkpoint + read_name + '.pth', map_location=device)
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict)
        logging.info(f'Model loaded from ' + read_name + '.pth')

    net = convert_model(net)
    net = torch.nn.parallel.DataParallel(net.to(device))
    torch.backends.cudnn.benchmark = True
    train_net(read_name=args.read_name,
              save_name=args.save_name,
              DataSet=args.DataSet,
              net=net,
              device=device,
              data_path=args.data_path,
              epochs=args.epochs,
              batch_size=args.batch_size,
              lr=args.lr,
              num_workers=args.numworkers
              )


if __name__ == '__main__':
    args = parse_args()
    dir_checkpoint = 'save_weights/'
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
