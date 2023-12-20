#!/usr/bin/python2.7
import os
import sys
import json
import torch
import random

from shutil import copy2

from util.utils import create_argument_parser, parse_config

from loaders.data_loader_temp import BatchGenerator
from experiments.lstm import LSTMTrainer
from experiments.mtms_tcn import TCNTrainer

import logging
logging.basicConfig(format='%(asctime)s %(levelname)-8s: %(message)s',
    stream=sys.stdout, level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')

def create_data_loaders(hp, exp_path, sample_rate=1):
    train_batch_gen = BatchGenerator(
                        hp.n_phases, hp.n_steps, os.path.join(exp_path, 'train_videos.pickle'),
                        hp.num_sup_videos, hp.num_weak_sup_vids, sample_rate=sample_rate
                    )

    valid_batch_gen = BatchGenerator(
                        hp.n_phases, hp.n_steps, os.path.join(exp_path, 'valid_videos.pickle'), sample_rate=sample_rate
                    )
    test_batch_gen  = BatchGenerator(
                        hp.n_phases, hp.n_steps, os.path.join(exp_path, 'test_videos.pickle'), sample_rate=sample_rate
                    )

    return train_batch_gen, valid_batch_gen, test_batch_gen

def Main(args):
    logging.info('\n'*5)
    logging.info('-'*50)
    logging.info('Module: LSTM/BiLSTM-CRF/TCN'.center(50))
    logging.info('-'*50)
    
    hp_file = args.hyper_params
    hp = parse_config(hp_file).default
    
    exp_path = os.path.join(args.path, hp.run_dir)

    stats = {}
    stats_fp = '{}/{}/{}'.format(args.path, hp.pkl_path, 'stats_test.json')
    if os.path.exists(stats_fp): stats = json.load(open(stats_fp, 'r'))
    stats_fp = '{}/{}/{}'.format(args.path, hp.pkl_path, 'stats.json')
    if os.path.exists(stats_fp): stats = json.load(open(stats_fp, 'r'))
    
    stats['experiment'] = exp_path
    stats['preload_model'] = args.load_model
    stats['state'] = args.state
    stats['dataset'] = os.path.join(args.path, hp.pkl_path) if hp.read_pkl else args.dataset
    stats['model_path'] = exp_path
    stats['best_model'] = 0 # best model at epoch 0 (default)
    stats['latest_epoch'] = -1 # latest model at epoch -1 (default)
    stats['best_m_stats'] = [-1, 0.0, 0.0]
    stats['experiment_stats'] = []

    if not os.path.exists(stats['model_path']):
        os.makedirs(stats['model_path'])

    copy2(hp_file, stats['model_path'])

    sample_rate = 1
    device = torch.device("cuda:" + str(hp.gpu_devices[0]))

    seed =  hp.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if args.state == 'extract_predictions':
        hp.num_weak_sup_vids = -1

    Trainer = LSTMTrainer if 'LSTM' in hp.method else TCNTrainer
    train_batch_gen, valid_batch_gen, test_batch_gen = create_data_loaders(hp, stats['dataset'], sample_rate)
    trainer = Trainer(stats, hp)
    trainer.resume_training()
    
    if args.state == "train":
        stats = trainer.train(
            train_batch_gen, valid_batch_gen, device
        )
    elif args.state == "extract_predictions":
        trainer.predict(
            train_batch_gen, valid_batch_gen, test_batch_gen,
            device, sample_rate
        )
    return

if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()

    Main(args)
