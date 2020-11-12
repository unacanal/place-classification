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
### Using Places365 indoor test dataset

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

    #----------------- load weights ---------------
    weights = []
    weight_f = open('./data/csv/weighted_labels_outdoor_5_places365.csv', 'r')
    weight_reader = csv.reader(weight_f)
    for row in weight_reader:
        weights.append(row)
    weight_f.close()
    #------------------ load GT --------------------
    reader = csv.reader(open('dataset/MITPlaces_indoor/gt_MITPlaces_4_test.csv', 'r'))
    gt = {}
    for row in reader:
        k, v = row
        gt[k] = v
    # --------------------------------------------------------------------------------
    img_files = []
    for path, label in gt.items():
        img_files.append("./dataset/MITPlaces_indoor/" + path)
    num_files = len(img_files)

    correct = 0
    attic_wrong = 0
    bathroom_wrong = 0
    bedroom_wrong = 0
    kitchen_wrong = 0

    indoor_labels1 = [62, 63, 65, 65, 66, 67, 68, 69, 70, 71]  # things-indoor-furnitures: door, toilet, desk(69x), window(68x), dining table, mirror(66x), bed, potted plant, couch, chair
    indoor_labels2 = [123,161,130,107,108,98,156,165] # stuff-indoor-furniture: furniture-other, stairs, light, counter, mirror(66x), cupboard, cabinet, shelf, table, desk(69x), door(71x)
    indoor_labels = [62, 63, 65, 65, 66, 67, 68, 69, 70, 71,123,161,130,107,108,98,156,165]
    for img_file in img_files:
        # Inference
        weight_sum = [0, 0, 0, 0]  # [attic, bathroom, bedroom, kitchen]
        print(img_file)
        img_file_name = img_file.split('/')[3]  # file name without path
        img_file = img_file.replace("\\", "/")
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image, raw_image = preprocessing(image, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        labels = np.unique(labelmap)

        for label in labels:
            if label in indoor_labels:  # if the label is 'indoor_labels'
                w = 4
            else:
                w = 1
            for i in range(4):
                weight_sum[i] += (w*float(weights[label][i]))
        print(weight_sum, np.argmax(weight_sum) + 1, gt[img_file_name]) ## gt: 1~4, argmax: 0~
        if int(gt[img_file_name]) == np.argmax(weight_sum) + 1:
            correct += 1
            print("correct")
        else:
            gtgt = int(gt[img_file_name])
            print("wrong")
            if gtgt == 1:
                attic_wrong += 1
            if gtgt == 2:
                bathroom_wrong += 1
            if gtgt == 3:
                bedroom_wrong += 1
            if gtgt == 4:
                kitchen_wrong += 1

    accuracy = correct / num_files * 100
    print(correct, num_files)
    print(attic_wrong, bathroom_wrong, bedroom_wrong, kitchen_wrong)
    print(accuracy)

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
