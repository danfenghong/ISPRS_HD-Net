import os

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
batchsize = 16
num_workers = 0
read_name = 'HDNet_Mass_dice_best'
Dataset = 'Mass'
assert Dataset in ['WHU', 'Inria', 'Mass']
net = HighResolutionDecoupledNet(base_channel=48, num_classes=1)
print(sum(p.numel() for p in net.parameters()))


def eval_HRBR(net,
             device,
             batch_size):
    testdataset = BuildingDataset(dataset_dir=data_dir, training=False, txt_name="test.txt", data_name="Mass")
    test_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             drop_last=False)
    best_score = eval_net(net, test_loader, device, savename=Dataset + '_' + read_name)  #
    print('Best iou:', best_score)


data_dir = "/media/bimeiqiao/sda11/liyuxuan/data/Massachusetts/"
dir_checkpoint = 'save_weights/'
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    if read_name != '':
        net_state_dict = net.state_dict()
        state_dict = torch.load(dir_checkpoint + read_name + '.pth', map_location=device)
        net_state_dict.update(state_dict)
        net.load_state_dict(net_state_dict, strict=False)  # 删除了down1-3
        logging.info(f'Model loaded from ' + read_name + '.pth')

    net = convert_model(net)
    net = torch.nn.parallel.DataParallel(net.to(device))
    torch.backends.cudnn.benchmark = True
    eval_HRBR(net=net,
             batch_size=batchsize,
             device=device)
