import os
import sys
import cv2
import torch
import argparse
import subprocess
import numpy as np
from glob import glob
from PIL import Image
import os.path as osp
import scipy.io as io
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, distance_transform_cdt

parser = argparse.ArgumentParser()
parser.add_argument("--datadir", dest='datadir', default='/media/bimeiqiao/sda11/liyuxuan/data/Inria/')
parser.add_argument("--outname", default='boundary')
# parser.add_argument('--split', nargs='+', default=['train', 'test'])
parser.add_argument('--metric', default='euc', choices=['euc', 'taxicab'])
args = parser.parse_args()

label_list = [0, 255]

def _encode_label(labelmap):
    encoded_labelmap = np.ones_like(labelmap, dtype=np.uint16) * 255
    for i, class_id in enumerate(label_list):
        encoded_labelmap[labelmap == class_id] = i

    return encoded_labelmap


def process(inp):
    """
    segfix.lib.datasets.preprocess.cityscapes.dt_offset_generator.py
    """
    (indir, outdir, basename) = inp
    print(inp)
    labelmap = np.array(Image.open(osp.join(indir, basename)).convert("P")).astype(np.int16)/255
    a = np.sum(labelmap == 0)
    labelmap = _encode_label(labelmap)
    labelmap = labelmap + 1
    depth_map = np.zeros(labelmap.shape, dtype=np.float32)  # (H,W,C)

    # for id in range(1, len(label_list)):
    for id in range(1, len(label_list) + 1):  # only consider the outer boundary
        labelmap_i = labelmap.copy()
        labelmap_i[labelmap_i != id] = 0
        labelmap_i[labelmap_i == id] = 1
        # label: 1 background: 0
        if args.metric == 'euc':
            depth_i = distance_transform_edt(labelmap_i)
        elif args.metric == 'taxicab':
            depth_i = distance_transform_cdt(labelmap_i, metric='taxicab')
        else:
            raise RuntimeError
        depth_map += depth_i

    depth_map[depth_map > 250] = 250
    if a == (labelmap.shape[0] * labelmap.shape[1]):
        depth_map[depth_map > 0] = 250

    depth_map = depth_map.astype(np.uint8)

    # edge = depth_map.copy()
    # e1 = (depth_map > 0) & (depth_map <= 3)
    # edge[e1] = 255
    # e2 = (edge > 0) & (edge < 255)
    # edge[e2] = 0
    # ed = Image.fromarray(edge)
    # ed.save("edge.png")

    io.savemat(
        osp.join(outdir, basename.replace("tif", "mat")),
        {"depth": depth_map},
        do_compression=True,
    )


indir = osp.join(args.datadir, 'train', 'label')
outdir = osp.join(args.datadir, args.outname)
args_to_apply = [(indir, outdir, osp.basename(basename)) for basename in glob(osp.join(indir, "*.tif"))]
for i in tqdm(range(len(args_to_apply))):
    process(args_to_apply[i])
