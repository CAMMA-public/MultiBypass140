import argparse
import numpy as np
import torch.nn as nn

import yaml
from easydict import EasyDict as edict

## Median weighting
def compute_class_weights(classes, n_classes=13):
    labels, counts = np.unique(classes, return_counts=True)
    median = np.median(counts)
    weights = median/counts
    exp_labels = np.array(range(n_classes))

    missing = [idx for idx in exp_labels if idx not in labels]
    for miss in missing:
        weights = np.insert(weights, miss, 1.0)

    return weights.tolist()


def parse_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
    return config


def init_classification_loss(
        name='cross_entropy', weights=None, gamma=0.7, ignore_index=-100
    ):
    assert np.any([name == 'cross_entropy', name == 'focal_loss']), f'loss function {name} not implemented!'
    if name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
    elif name == 'focal_loss':
        raise Exception('Focal loss not implemented!')
    return loss
#------------------------------------------------------------------------------------------
# Argument parser
def create_argument_parser():
    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('-d', '--dataset', default='',
                    help='Dataset to use for training/testing.\n      Options: (/mnt/data/bypass40/)')

    parser.add_argument('-s', '--state', default='train',
                    help='State of the model.\n      Options: train, test, features')

    parser.add_argument('-lm', '--load_model', default='models/resnet_v2_50_2017_04_14/resnet_v2_50.ckpt',
                    help='path to model file to be loaded during initialization.\n')

    parser.add_argument('-pt', '--pretrain', default='True',
                    help='Loading pretrained model of resnet.\n      Options: True, False')

    parser.add_argument('-hp', '--hyper_params', default='hparams/hp_225_bypass.yaml',
                    help='Path to the hyper parameters config file.\n')

    parser.add_argument('-e', '--experiment_name', default='cnn_exp',
                    help='Name of the experiment.\n')

    parser.add_argument('-p', '--path', default='models',
                    help='Path tp save the experiment.\n')

    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
    return parser
