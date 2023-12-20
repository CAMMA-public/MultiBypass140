import os
import sys

import numpy as np
from shutil import copy2

from experiments.cnn import *
from loaders.data_loader_cnn import *

from util.utils import create_argument_parser, parse_config

import torch

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s: %(message)s',
    stream=sys.stdout, level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')

def Main(args):
    logging.info('\n'*5)
    logging.info('-'*50)
    logging.info('Module: CNN'.center(50))
    logging.info('-'*50)
    hp_file = args.hyper_params
    hp = parse_config(hp_file).default

    stats = {'experiment': hp.run_dir}
    stats['state'] = args.state
    stats['dataset'] = hp.data_dir if args.dataset == '' else args.dataset
    stats['best_model'] = 0 # best model at epoch 0 (default)
    stats['latest_epoch'] = -1 # latest model at epoch -1 (default)
    stats['best_m_stats'] = [-1, 0.0, 0.0]
    stats['preload_model'] = args.load_model
    stats['experiment_stats'] = []
    path = args.path
    stats['model_path'] = os.path.join(path, stats['experiment'])

    if not os.path.exists(stats['model_path']):
        os.makedirs(stats['model_path'])

    copy2(hp_file, stats['model_path'])
    
    seed = hp.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    shuffle = True
    augment = True

    ### If 'extracted_features' then do it on the full dataset
    if args.state == 'extract_predictions':
        hp.num_weak_sup_vids = -1
        shuffle = False
        
    data_loader_fn = create_data_loaders
    assert ('bern' in stats['dataset']) or ('strasbourg' in stats['dataset']), \
            f'Only bern or strasbourg data is accepted (Given {stats["dataset"]})'

    train_loader, valid_loader, test_loader = data_loader_fn(stats, hp, shuffle=shuffle, augment=augment)
    mtrainer = CNNTrainer
    trainer = mtrainer(stats, hp)
    trainer.build_model()
    trainer.add_data_loaders(train_loader, valid_loader, test_loader)

    trainer.resume_training()
    if args.state == "train":
        stats = trainer.train()    
    elif args.state == "extract_predictions":
        trainer.test()

    return

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    Main(args)
