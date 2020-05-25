#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import yaml
from addict import Dict

# ---- added for writing csv
import os
import datetime
import csv
import operator
# ---------------------------

import glob

# sort dict
from operator import itemgetter

from libs.models import *
from libs.utils import DenseCRF


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1]
            # classes[int(label[0])] = label[1].split(",")[0]
    return classes

def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)
    return labelmap

@click.group()
@click.pass_context
def main(ctx):
    """
    Demo with a trained model
    """

    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-i",
    "--image-path",
    type=click.Path(exists=True),
    required=True,
    help="Image to be processed",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
def single(config_path, model_path, image_path, cuda, crf):
    """
    Inference from multiple images
    """

    # Setup
    CONFIG = Dict(yaml.load(config_path))
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None
    print(classes)
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # ------------------------------------------------------------------------------
    # load place gt for images
    reader = csv.reader(open('./data/csv/gt5000.csv', 'r'))
    gt = {}
    for row in reader:
        k, v = row
        gt[k] = v

    # final output array
    weighted_labels = np.zeros((182, 3))
    #--------------------------------------------------------------------------------
    img_files = glob.glob("./dataset/images/val2017/*.jpg")

    outdoor_labels = [ 94,  95,  96,  97,  99, 106, 111, 113, 119, 120,
                      124, 125, 126, 127, 128, 129, 134, 135, 136, 138,
                      140, 142, 144, 145, 146, 147, 148, 149, 150, 151,
                      154, 155, 157, 158, 159, 160, 162, 163, 164, 166,
                      169, 178, 179, 182 ]

    outdoor_labels_5 = [94, 95, 96, 97, 111, 119, 120,
                       124, 125, 126, 127, 128, 129, 134, 135, 136,
                       140, 142, 144, 145, 147, 148, 149, 150, 151,
                       154, 155, 158, 159, 160, 162, 163, 166,
                       169, 178, 179, 182] # water, ground, solid, plant, building

    outdoor_labels_3 = [ 95, 96, 111, 125, 126, 127, 128, 135, 136,
                       140, 144, 145, 147, 149, 150, 151, 154, 158,
                       159, 160, 162, 166, 182] # ground, solid, building 23
    for img_file in img_files:
        # Inference
        print(img_file)
        img_file_name = img_file.split('/')[4] # file name without path
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image, raw_image = preprocessing(image, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        labels = np.unique(labelmap)

        image_label = int(gt[img_file_name]) # place label of an image

        for label in labels:
            print(label)
            # print(classes[label])
            # print(list(classes.keys())[list(classes.values()).index(label)])
            if label in outdoor_labels_3:  # if the label is in 'outdoor' labels
                w = 2
            else:
                w = 1
            if image_label == 0: # image label == indoor
                weighted_labels[label][0] += w
            elif image_label == 1: # == nature
                weighted_labels[label][1] += w
            else: # == city
                weighted_labels[label][2] += w

    for i in range(len(weighted_labels)):
        total = sum(weighted_labels[i])
        print(total)
        if total: # not 0
            for j in range(len(weighted_labels[i])):
                weighted_labels[i][j] /= total
    print(weighted_labels)
    np.savetxt('./data/csv/weighted_labels_outdoor_3_w2.csv', weighted_labels, delimiter=",", fmt='%f')

@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
@click.option("--camera-id", type=int, default=0, show_default=True, help="Device ID")
def live(config_path, model_path, cuda, crf, camera_id):
    pass


if __name__ == "__main__":
    main()
