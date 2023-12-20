import os
import json
import time
import logging
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from util.utils import init_classification_loss


class LSTM(nn.Module):

    def __init__(
            self, embedding_dim, num_f_maps, num_phases, num_steps,
            phases=True, steps=False, bidir=False, dropout=0.0
        ):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_f_maps = num_f_maps
        self.feat_dims = num_f_maps
        self.phases = phases
        self.steps = steps
        self.num_phases = num_phases
        self.num_steps = num_steps
        self.feat_dims = num_f_maps
        if bidir:
            self.feat_dims = 2 * num_f_maps

        self.lstm = nn.LSTM(embedding_dim, num_f_maps, batch_first=True,
                            dropout=dropout, bidirectional=bidir)

        # Maps the output of the LSTM into tag space.
        if self.phases: self.phase_fc = nn.Linear(self.feat_dims, self.num_phases )
        if self.steps: self.step_fc = nn.Linear(self.feat_dims, self.num_steps )

    def _get_lstm_features(self, x, mask):
        pred_phases, pred_steps = [], []
        self.lstm.flatten_parameters()
        x = x.transpose(2, 1)
        out, _ = self.lstm(x)
        if self.phases:
            pred_phases = self.phase_fc(out).transpose(2, 1) * mask[:, 0:1, :]
            pred_phases = pred_phases.unsqueeze(0)
        if self.steps:
            pred_steps = self.step_fc(out).transpose(2, 1) * mask[:, 0:1, :]
            pred_steps = pred_steps.unsqueeze(0)
        return out, pred_phases, pred_steps
    
    def forward(self, x, mask):
        feats, pred_phases, pred_steps = self._get_lstm_features(x, mask)
        return feats, pred_phases, pred_steps

class LSTMTrainer:
    def __init__(self, stats, hp):
        self.stats = stats
        self.hp = hp
        self.num_phases = hp.n_phases
        self.num_steps = hp.n_steps
        self.add_phases = hp.add_phases
        self.add_steps = hp.add_steps

        model = LSTM

        self.model = model(
            hp.n_cnn_outputs, hp.num_f_maps,
            self.num_phases, self.num_steps,
            phases=self.add_phases, steps=self.add_steps
        )

        loss_name = hp.loss_type #'cross_entropy' 
        self.ce_ph = init_classification_loss(
            name=loss_name,
            weights=torch.FloatTensor(stats['phase_dist_train']) if hp.phases_weighted else None,
            gamma=hp.focal_gamma,
            ignore_index=-100
        )
        self.ce_st = init_classification_loss(
            name=loss_name,
            weights=torch.FloatTensor(stats['step_dist_train']) if hp.steps_weighted else None,
            gamma=hp.focal_gamma,
            ignore_index=-100
        )
        
        self.model.cuda()
        learning_rate = self.hp.temporal_lr
        if type(learning_rate) is str: learning_rate = float(learning_rate)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.weights_init = self.stats['model_path'].replace(self.hp.run_dir, self.hp.weights_init)

        logging.info('Train phases         : ' + str(self.add_phases))
        logging.info('Train steps          : ' + str(self.add_steps))
        logging.info('Learning rate        : ' + str(learning_rate))
        logging.info('Loading model        : ' + str(self.weights_init))
        logging.info('Saving model         : ' + str(self.stats['model_path']))


    def load_model(self, model_file, pretrain=False):
        if pretrain: return
        logging.info('Loading pretrained model: ' + str(model_file))
        self.model.load_state_dict(torch.load(model_file), strict=False)
        self.optimizer.load_state_dict(torch.load(model_file.replace('.model', '.opt')))
        return
    
    def resume_training(self):
        logging.info('-'*50)
        json_file = os.path.join(self.weights_init, 'stats_lstm.json')
        if self.hp.resume_training and os.path.isfile(json_file):
            new_dataset = self.stats['dataset']
            new_model_path = self.stats['model_path'] 
            stats_prev = json.load(open(json_file, 'r'))
            self.stats = {**self.stats, **stats_prev}
            self.stats['dataset'] = new_dataset
            self.stats['model_path'] = new_model_path
            logging.info('Resuming training from epoch: %d !!!'%self.stats['latest_epoch'])
            self.load_model(os.path.join(self.weights_init, 'epoch-latest_tm.model'))
            logging.info(f'Model path for saving run log: {self.stats["model_path"]}')
        else:
            logging.info('Resume training: False !!!')
        logging.info('-'*50)
        return
    
    def train(self, train_batch_gen, valid_batch_gen, device):
        self.model.train()
        exp_dir = self.stats['model_path']
        num_epochs, batch_size = self.hp.n_epochs_temp_model, self.hp.n_batch
        self.ce_ph.to(device)
        self.ce_st.to(device)
        
        best_m, best_m_corr_ph, best_m_corr_st = self.stats['best_m_stats']
        start_epoch = self.stats['latest_epoch'] + 1
        logging.info('Starting training from epoch %d/%d' % (start_epoch, num_epochs))

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            correct_ph, correct_st = 0, 0
            total = 0
            while train_batch_gen.has_next():
                _, batch_input, batch_phases, batch_steps, mask, supervised = train_batch_gen.next_batch(batch_size)
                batch_input, batch_phases, batch_steps, mask = batch_input.to(device), batch_phases.to(device), batch_steps.to(device), mask.to(device)
                self.optimizer.zero_grad()
                feats, pred_phases, pred_steps = self.model(batch_input, mask)

                loss = 0
                if self.add_phases:
                    for p in pred_phases:
                        loss += self.ce_ph(p.transpose(2, 1).contiguous().view(-1, self.num_phases), batch_phases.view(-1))

                    t, predicted = torch.max(pred_phases[-1].data, 1)
                    correct_ph += ((predicted == batch_phases).float()*mask[:, 0, :].squeeze(1)).sum().item()
                
                if self.add_steps:
                    for p in pred_steps:
                        loss += self.ce_st(p.transpose(2, 1).contiguous().view(-1, self.num_steps), batch_steps.view(-1))
                    _, predicted = torch.max(pred_steps[-1].data, 1)
                    correct_st += ((predicted == batch_steps).float()*mask[:, 0, :].squeeze(1)).sum().item()

                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                total += torch.sum(mask[:, 0, :]).item()

            valid_loss = 0
            valid_corr_ph, valid_corr_st = 0, 0
            valid_total = 0
            while valid_batch_gen.has_next():
                _, batch_input, batch_phases, batch_steps, mask, supervised = valid_batch_gen.next_batch(batch_size)
                batch_input, batch_phases, batch_steps, mask = batch_input.to(device), batch_phases.to(device), batch_steps.to(device), mask.to(device)
                self.optimizer.zero_grad()
                feats, pred_phases, pred_steps = self.model(batch_input, mask)

                loss = 0
                if self.add_phases:
                    for p in pred_phases:
                        loss += self.ce_ph(p.transpose(2, 1).contiguous().view(-1, self.num_phases), batch_phases.view(-1))

                    _, predicted = torch.max(pred_phases[-1].data, 1)
                    valid_corr_ph += ((predicted == batch_phases).float()*mask[:, 0, :].squeeze(1)).sum().item()

                if self.add_steps:
                    for p in pred_steps:
                        loss += self.ce_st(p.transpose(2, 1).contiguous().view(-1, self.num_steps), batch_steps.view(-1))

                    _, predicted = torch.max(pred_steps[-1].data, 1)
                    valid_corr_st += ((predicted == batch_steps).float()*mask[:, 0, :].squeeze(1)).sum().item()

                valid_loss += loss.item()
                valid_total += torch.sum(mask[:, 0, :]).item()

            train_batch_gen.reset()
            valid_batch_gen.reset()

            epoch_loss /= len(train_batch_gen.videos)
            valid_loss /= len(valid_batch_gen.videos)
            correct_ph = float(correct_ph)/total
            correct_st = float(correct_st)/total
            valid_corr_ph = float(valid_corr_ph)/valid_total
            valid_corr_st = float(valid_corr_st)/valid_total

            self.stats['experiment_stats'].append([
                epoch, epoch_loss, valid_loss, correct_ph, correct_st,
                valid_corr_ph, valid_corr_st, time.time()
            ])

            if valid_corr_ph + valid_corr_st >= best_m_corr_ph + best_m_corr_st:
                best_m = 'best_tm'
                best_m_corr_ph = valid_corr_ph
                best_m_corr_st = valid_corr_st
                self.stats['best_model'] = best_m
                self.stats['best_m_stats'] = [epoch, best_m_corr_ph, best_m_corr_st]

                torch.save(self.model.state_dict(), exp_dir + "/epoch-" + str(best_m) + ".model")
                torch.save(self.optimizer.state_dict(), exp_dir + "/epoch-" + str(best_m) + ".opt")

                logging.info("[best epoch %d]: epoch loss = %f, valid loss = %f, acc ph = %f, acc st = %f,  valid acc ph = %f, valid acc st = %f" % (
                        self.stats['experiment_stats'][-1][0], self.stats['experiment_stats'][-1][1],
                        self.stats['experiment_stats'][-1][2], self.stats['experiment_stats'][-1][3],
                        self.stats['experiment_stats'][-1][4], self.stats['experiment_stats'][-1][5],
                        self.stats['experiment_stats'][-1][6]
                    )
                )

            logging.info("[epoch %d]: epoch loss = %f, valid loss = %f, acc ph = %f, acc st = %f,  valid acc ph = %f, valid acc st = %f" % (
                    self.stats['experiment_stats'][-1][0], self.stats['experiment_stats'][-1][1],
                    self.stats['experiment_stats'][-1][2], self.stats['experiment_stats'][-1][3],
                    self.stats['experiment_stats'][-1][4], self.stats['experiment_stats'][-1][5],
                    self.stats['experiment_stats'][-1][6]
                )
            )
            
            self.stats['latest_epoch'] = epoch
            torch.save(self.model.state_dict(), exp_dir + "/epoch-latest_tm" + ".model")
            torch.save(self.optimizer.state_dict(), exp_dir + "/epoch-latest_tm" + ".opt")
            
            ## Save training stats after each epoch
            with open(os.path.join(exp_dir, 'stats_lstm.json'), 'w') as fp:
                json.dump(self.stats, fp)

        return self.stats

    def predict(
        self, train_batch_gen, valid_batch_gen, test_batch_gen, device, sample_rate
    ):
        model_dir = self.stats['model_path']
        batch_size = self.hp.n_batch
        epoch = self.stats['best_model']

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(f'{self.weights_init}/epoch-{str(epoch)}.model'))
            self.ce_ph.to(device)
            self.ce_st.to(device)

            dsets = ['train', 'valid', 'test']
            datasets = [train_batch_gen, valid_batch_gen, test_batch_gen]

            for i, ds in enumerate(dsets):
                phase_preds = []
                phase_labels = []
                step_preds = []
                step_labels = []
                test_loss = 0
                total = 0
                images = []
                
                while datasets[i].has_next():
                    batch_images, batch_input, batch_phases, batch_steps, mask, supervised = datasets[i].next_batch(batch_size)
                    batch_input, batch_phases, batch_steps, mask = batch_input.to(device), batch_phases.to(device), batch_steps.to(device), mask.to(device)
                    feats, pred_phases, pred_steps = self.model(batch_input, mask)

                    images.append(batch_images[0][:])
                    loss = 0
                    if self.add_phases:
                        for p in pred_phases:
                            loss += self.ce_ph(p.transpose(2, 1).contiguous().view(-1, self.num_phases), batch_phases.view(-1))
                        
                        _, predicted = torch.max(pred_phases[-1].data, 1)
                        phase_preds.append(predicted.detach().cpu().numpy().squeeze())
                        phase_labels.append(batch_phases.detach().cpu().numpy().squeeze())
                    
                    if self.add_steps:
                        for p in pred_steps:
                            loss += self.ce_st(p.transpose(2, 1).contiguous().view(-1, self.num_steps), batch_steps.view(-1))

                        _, predicted = torch.max(pred_steps[-1].data, 1)
                        step_preds.append(predicted.detach().cpu().numpy().squeeze())
                        step_labels.append(batch_steps.detach().cpu().numpy().squeeze())

                    test_loss += loss.item()
                    total += torch.sum(mask[:, 0, :]).item()

                test_loss /= len(datasets[i].videos)
                self.stats[ds + '_loss_tm'] = test_loss
                with open(os.path.join(model_dir, ds + '_imgs.npy'), 'wb') as fp:
                    np.save(fp, np.concatenate(images))
                if self.add_phases:
                    with open(os.path.join(model_dir, ds + '_phase_labels.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate(phase_labels))
                    with open(os.path.join(model_dir, ds + '_phase_preds.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate(phase_preds))
                if self.add_steps:
                    with open(os.path.join(model_dir, ds + '_step_labels.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate(step_labels))
                    with open(os.path.join(model_dir, ds + '_step_preds.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate(step_preds))

                ## Save training stats after each epoch
                with open(os.path.join(model_dir, 'stats_lstm.json'), 'w') as fp:
                    json.dump(self.stats, fp)
