import torch
import random
import pickle
import logging
import numpy as np

class BatchGenerator(object):
    def __init__(self, num_phases, num_steps, datafile, sample_rate=1
        ):
        self.index = 0
        self.videos = list()
        self.num_phases = num_phases
        self.num_steps = num_steps
        self.datafile = datafile
        self.sample_rate = sample_rate
        self._read_data()

    def reset(self):
        self.index = 0
        random.shuffle(self.videos)

    def has_next(self):
        if self.index < len(self.videos):
            return True
        return False

    def next_batch(self, batch_size):
        video_batch = self.videos[self.index:self.index + batch_size]
        self.index += batch_size
        bsize = 1e5
        batch_input = []
        batch_phases = []
        batch_steps = []
        batch_images = []
        supervision_signals = []
        for video in video_batch:
            features = np.array(video['features']).T
            phases = video['phase_labels']
            im_files = video['images']
            try:
                steps = video['step_labels']
            except:
                steps = video['phase_labels']
            supervision_signals.append(video['supervision'])

            seq_len = len(im_files)
            batch_images += np.array_split(im_files[:], seq_len // bsize + 1)
            batch_input += np.array_split(features[::self.sample_rate], seq_len // bsize + 1, 1)
            batch_phases += np.array_split(phases[::self.sample_rate], seq_len // bsize + 1)
            batch_steps += np.array_split(steps[::self.sample_rate], seq_len // bsize + 1)


        max_seq = np.shape(batch_input[0])[1]
        
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], np.shape(batch_input[0])[1], dtype=torch.float)
        batch_phases_tensor = torch.ones(len(batch_input), max_seq, dtype=torch.long)*(-100)
        batch_steps_tensor = torch.ones(len(batch_input), max_seq, dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_phases, max_seq, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_phases_tensor[i, :np.shape(batch_phases[i])[0]] = torch.from_numpy(batch_phases[i])
            batch_steps_tensor[i, :np.shape(batch_steps[i])[0]] = torch.from_numpy(batch_steps[i])
            mask[i, :, :np.shape(batch_phases[i])[0]] = torch.ones(self.num_phases, np.shape(batch_phases[i])[0])

        return batch_images, batch_input_tensor, batch_phases_tensor, batch_steps_tensor, mask, supervision_signals

    def _read_data(self):
        with open(self.datafile, 'rb') as fp:
            self.videos = pickle.load(fp)
        
        random.shuffle(self.videos)
        logging.info('-' * 50)
        logging.info('datafile                   : ' + self.datafile)
        logging.info('No of videos loaded        : ' + str(len(self.videos)))
        logging.info('-' * 50)
