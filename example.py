import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import test
import argparse

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='teedy/UCF-Train-Val-Test',
                        help='training data directory')
    parser.add_argument('--save-dir', default='teedy/vgg/0106-190713',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    test.density(args)

