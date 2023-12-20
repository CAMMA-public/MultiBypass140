import os
import random
import logging
import numpy as np

from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from util.utils import compute_class_weights
from util.data_reader import read_pkl_data

class Bypass40Dataset(Dataset):
    '''Data loader for Bypass40 dataset'''

    def __init__(
            self, videos, phase_labels, step_labels, augment='', transform=None
        ):
        self.videos = videos
        self.phase_labels = phase_labels
        self.step_labels = step_labels
        self.transform = transform
        self.augment = augment
        self.num_sup_videos = len(videos)        
        self.step_sup_vid_mask = [0] * len(self.videos)
        self.indices = list(range(len(self.videos)))
        self.video_lengths = list(map(len, self.videos))
        self.total_imgs = sum(self.video_lengths)

        logging.info('videos               : %d'%len(self.videos))
        logging.info('augment              : %s'%str(self.augment))

    def __len__(self):
        return self.total_imgs

    def shuffle_videos(self):
        random.shuffle(self.indices)
        return True

    def find_video_image_indices(self, idx):
        varr = idx - np.cumsum([0]+[self.video_lengths[j] for j in self.indices])
        varr_idx = np.where(varr < 0)[0][0] - 1
        vidx = self.indices[varr_idx]
        im_idx = varr[varr_idx]

        video_end = 0
        if im_idx == self.video_lengths[vidx] - 1:
            video_end = 1

        video_sup = self.step_sup_vid_mask[vidx]
        return vidx, im_idx, video_sup, video_end

    def __getitem__(self, idx):
        vidx, im_idx, video_sup, video_end = self.find_video_image_indices(idx)

        img_path = self.videos[vidx][im_idx]
        image = Image.open(self.videos[vidx][im_idx])
        phase_label = self.phase_labels[vidx][im_idx]
        step_label = self.step_labels[vidx][im_idx]

        if self.augment != 'original':
            img_path = img_path.split('/')
            img_path[-2] = ''.join([self.augment, '-', img_path[-2]])
            img_path = '/'.join(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return img_path, image, phase_label, step_label, video_sup, video_end

    def class_distribution(self, n_phases=13, n_steps=46):
        phase_dist = compute_class_weights(sum(self.phase_labels,[]), n_phases)
        step_dist = compute_class_weights(
            sum([self.step_labels[i] for i, v in enumerate(self.step_sup_vid_mask) if v == 1],[]),
            n_steps
        )
        return phase_dist, step_dist

def get_augment_transforms(hp):
    aug_transforms = {
        'original': transforms.Compose([
            transforms.Resize((hp.n_resize, hp.n_resize)),
            transforms.ToTensor(),
        ])
    }

    for aug in hp.augment_funcs:
        aug_transforms[aug] = transforms.Compose(sum([
                [transforms.Resize((hp.n_resize, hp.n_resize))],
                [transforms.RandomHorizontalFlip(1)] if aug == 'flip' else [],
                [transforms.RandomRotation((hp.aug_rot_angle, hp.aug_rot_angle))] if aug == 'rotate' else [],
                [transforms.ColorJitter(saturation=[hp.saturate_level, hp.saturate_level])] if aug == 'saturate' else [],
                [transforms.ToTensor()],
            ], []))
    return aug_transforms


def create_data_loader(
        videos, phase_labels, step_labels, stats, hp, setname='train', augment=False
    ):
    vset = videos
    pset = phase_labels
    sset = step_labels
    aug_transforms = get_augment_transforms(hp)

    if not augment:
        aug_transforms = dict(filter(lambda elem: elem[0] == 'original', aug_transforms.items()))

    dataset = ConcatDataset([
            Bypass40Dataset(vset, pset, sset, aug, transform)
            for aug, transform in aug_transforms.items()
        ])
    
    num_set = len(dataset)
    minibatch = hp.n_minibatch
    phase_dist, step_dist = dataset.datasets[0].class_distribution(hp.n_phases, hp.n_steps)

    stats['num_'+setname] = num_set
    stats['num_'+setname+'_batch'] = int(np.ceil(num_set / minibatch))
    stats['videos_'+setname] = []
    stats['phase_dist_'+setname] = phase_dist
    stats['step_dist_'+setname] = step_dist
    logging.info(f'total video in {setname}: {len(videos)}')
    logging.info(' '.join(map(str, [setname, stats['num_'+setname], stats['num_'+setname+'_batch']])))
    return dataset

def create_data_loaders(stats, hp, shuffle=False, augment=True):
    read_pkl = hp.read_pkl if hasattr(hp, 'read_pkl') else False
    logging.info('Reading from pkl     : ' + str(read_pkl))

    n_workers = hp.n_parallel * torch.cuda.device_count()
    minibatch = hp.n_minibatch
    pin_memory = True

    ## Read train pickle files
    if 'bern_stras' in hp.dataset:
        dataset = stats['dataset']
        dataset_n = stats['dataset'].replace('bern', 'strasbourg')
    else:
        dataset, dataset_n = stats['dataset'], []
    labels = os.path.join(dataset, 'labels', 'train', f'1fps_100_{hp.id_split}.pickle')
    images = os.path.join(dataset, 'frames')
    print(labels, images)
    videos, phase_labels, step_labels = read_pkl_data(
        labels, images
    )
    if dataset_n != []:
        labels = os.path.join(dataset_n, 'labels', 'train', f'1fps_100_{hp.id_split}.pickle')
        images = os.path.join(dataset_n, 'frames')
        videos_n, phase_labels_n, step_labels_n = read_pkl_data(
            labels, images
        )
        videos += videos_n
        phase_labels += phase_labels_n
        step_labels += step_labels_n
    logging.info(f'two datasets: {dataset}, {dataset_n}')
    datasets = create_data_loader(
                    videos, phase_labels, step_labels, stats, hp,
                    setname='train', augment=augment
                )
    train_loader = DataLoader(
                    datasets, batch_size=minibatch,
                    shuffle=shuffle, num_workers=n_workers, drop_last=False,
                    pin_memory=pin_memory, prefetch_factor=hp.n_prefetch
                )

    ## Read val pickle files
    labels = os.path.join(dataset, 'labels', 'val', f'1fps_{hp.id_split}.pickle')
    images = os.path.join(dataset, 'frames')
    videos, phase_labels, step_labels = read_pkl_data(
        labels, images
    )
    if dataset_n != []:
        labels = os.path.join(dataset_n, 'labels', 'val', f'1fps_{hp.id_split}.pickle')
        images = os.path.join(dataset_n, 'frames')
        videos_n, phase_labels_n, step_labels_n = read_pkl_data(
            labels, images
        )
        videos += videos_n
        phase_labels += phase_labels_n
        step_labels += step_labels_n
        
    datasets = create_data_loader(
                    videos, phase_labels, step_labels,
                    stats, hp, setname='valid', augment=False
                )
    valid_loader = DataLoader(
                    datasets, batch_size=minibatch,
                    shuffle=shuffle, num_workers=n_workers, drop_last=False,
                    pin_memory=pin_memory, prefetch_factor=hp.n_prefetch
                )

    ## Read test pickle files
    labels = os.path.join(dataset, 'labels', 'test', f'1fps_{hp.id_split}.pickle')
    images = os.path.join(dataset, 'frames')
    videos, phase_labels, step_labels = read_pkl_data(
        labels, images
    )
    if dataset_n != []:
        labels = os.path.join(dataset_n, 'labels', 'test', f'1fps_{hp.id_split}.pickle')
        images = os.path.join(dataset_n, 'frames')
        videos_n, phase_labels_n, step_labels_n = read_pkl_data(
            labels, images
        )
        videos += videos_n
        phase_labels += phase_labels_n
        step_labels += step_labels_n

    datasets = create_data_loader(
                    videos, phase_labels, step_labels,
                    stats, hp, setname='test', augment=False
                )
    test_loader = DataLoader(
                    datasets, batch_size=minibatch,
                    shuffle=shuffle, num_workers=n_workers, drop_last=False,
                    pin_memory=pin_memory, prefetch_factor=hp.n_prefetch
                )

    return train_loader, valid_loader, test_loader
