import os
import time
import torch

import json
import pickle
import logging
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch.nn as nn
from torch import optim
import torchvision.models as models

from progress.bar import IncrementalBar
from util.utils import init_classification_loss

class MultiTaskResNet50(nn.Module):
    def __init__(self, hparams):
        super(MultiTaskResNet50, self).__init__()
        self.model = models.resnet50
        if not os.path.isfile(hparams.pretrain_model):
            self.model = self.model(pretrained=hparams.pretained_model)
        else:
            logging.info('Loading model from local!!!')
            pretrained_weights = torch.load(hparams.pretrain_model)
            self.model = self.model(pretrained=False)
            self.model.load_state_dict(pretrained_weights)

        # replace final layer with number of labels
        self.model.fc = Identity()
        self.fc_phase = nn.Linear(hparams.n_cnn_outputs, hparams.n_phases)
        self.fc_step = nn.Linear(hparams.n_cnn_outputs, hparams.n_steps)
        self.use_amp = hparams.use_amp

        logging.info('Backbone Model       : ' + hparams.model)
        logging.info('output dim           : ' + str(hparams.n_cnn_outputs))
        logging.info('phase fc dim         : ' + str(hparams.n_phases))
        logging.info('step fc dim          : ' + str(hparams.n_steps))
        logging.info('use_amp              : ' + str(hparams.use_amp))

    def forward(self, x, mask):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            out_stem = self.model(x)
            phase = self.fc_phase(out_stem)
            step = self.fc_step(out_stem)
        return out_stem, phase, step, None, None, None


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNNTrainer(object):

    def __init__(self, stats, hp):
        self.stats = stats
        self.hp = hp
        self.num_phases = hp.n_phases
        self.num_steps = hp.n_steps
        self.add_phases = hp.add_phases
        self.add_steps = hp.add_steps
        self.mini_batch = hp.n_minibatch

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
        self.mse = nn.MSELoss(reduction='none')
        self.weights_init = self.stats['model_path'].replace(self.hp.run_dir, self.hp.weights_init)

        logging.info('Train phases         : ' + str(self.add_phases))
        logging.info('Train steps          : ' + str(self.add_steps))
        logging.info('Training minibatch   : ' + str(self.mini_batch))
        logging.info('Learning rate        : ' + str(hp.learning_rate))
        logging.info('Loading model        : ' + str(self.weights_init))
        logging.info('Saving model         : ' + str(self.stats['model_path']))

    def build_model(self):
        self.model = MultiTaskResNet50(self.hp)
        self.use_amp = self.hp.use_amp # replace with - self.hp.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.setup_multi_gpu_model(multi_gpu=True)

    def setup_multi_gpu_model(self, multi_gpu=False):
        '''
            Multi-gpu training
        '''
        devices = [torch.device("cuda:" + str(gpu)) for gpu in self.hp.gpu_devices]
        self.model.train()

        self.model.cuda()
        self.ce_ph.cuda()
        self.ce_st.cuda()
        self.mse.cuda()

        if multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.loss_func = self.combined_loss
        self.optimizer = optim.Adam(
                            self.model.parameters(),
                            lr=self.hp.learning_rate,
                            weight_decay=self.hp.weight_decay
                        )

    def load_model(self, model_file, pretrain=False):
        if pretrain: return
        logging.info('Loading pretrained model: ' + str(model_file))
        self.model.load_state_dict(torch.load(model_file), strict=False)
        self.optimizer.load_state_dict(torch.load(model_file.replace('.model', '.opt')))
        if os.path.isfile(model_file.replace('.model', '.sclr')):
            self.scaler.load_state_dict(torch.load(model_file.replace('.model', '.sclr')))
        return

    def resume_training(self):
        logging.info('-'*50)
        json_file = os.path.join(self.weights_init, 'stats.json')
        if self.hp.resume_training and os.path.isfile(json_file):
            new_dataset = self.stats['dataset']
            new_model_path = self.stats['model_path'] 
            stats_prev = json.load(open(json_file, 'r'))
            self.stats = {**self.stats, **stats_prev}
            self.stats['dataset'] = new_dataset
            self.stats['model_path'] = new_model_path
            logging.info('Resuming training from epoch: %d !!!'%self.stats['latest_epoch'])
            self.load_model(os.path.join(self.weights_init, 'epoch-latest.model'))
            logging.info(f'Model path for saving run log: {self.stats["model_path"]}')
        else:
            logging.info('Resume training: False !!!')
        logging.info('-'*50)
        return

    def add_data_loaders(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

    def get_train_reg_params(self):
        train_params = []
        reg_params = []

        for layer in self.hp.regularize:
            for name, param in self.model.named_parameters():
                if layer in name:
                    reg_params.append(param)

        for layer in self.hp.train_vars:
            for name, param in self.model.named_parameters():
                if layer in name:
                    train_params.append(param)

        return reg_params, train_params

    def l2_weight_reg_loss(self):
        loss = torch.tensor(0.).cuda()
        for layer in self.hp.regularize:
            for name, param in self.model.named_parameters():
                if layer in name and 'weight' in name:
                    loss = loss + torch.norm(param) ** 2

        return loss

    def combined_loss(
            self, batch_phases, batch_steps,
            pred_phases, pred_steps, video_sup
        ):
        loss = 0.0
        if self.add_phases:
            loss += self.ce_ph(
                        pred_phases,
                        batch_phases.view(-1)
                    )

        if self.add_steps:
            nz = torch.nonzero(video_sup).numel() == self.mini_batch
            loss += int(nz) * self.ce_st(
                        pred_steps,
                        batch_steps.view(-1)
                    )

        return loss

    def accuracy(self, labels, predictions):
        t, predicted = torch.max(predictions.data, 1)
        acc = (predicted == labels).float().mean().item()
        return acc

    def train_epoch(self):
        previous_time = datetime.now()
        train_loss = 0.0
        train_ph_acc = 0.0
        train_st_acc = 0.0

        bar = IncrementalBar('Train', max = self.stats['num_train_batch'])
        for bidx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            _, batch_input, batch_phases, batch_steps, video_sup, _ = batch
            batch_input, batch_phases, batch_steps = batch_input.cuda(), batch_phases.cuda(), batch_steps.cuda()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                _, pred_phases, pred_steps, _, _, _ = self.model(batch_input, None)

                loss = self.loss_func(
                                batch_phases, batch_steps,
                                pred_phases, pred_steps, video_sup
                            )
            train_loss += loss.item()
            if self.add_phases: train_ph_acc += self.accuracy(batch_phases, pred_phases)
            if self.add_steps: train_st_acc += self.accuracy(batch_steps, pred_steps)

            self.scaler.scale(loss).backward()
            # # Unscales gradients and calls
            # # or skips optimizer.step()
            self.scaler.step(self.optimizer)
            # # Updates the scale for next iteration
            self.scaler.update()
            ## without amp and gradscaler
            # loss.backward()
            # self.optimizer.step()
            torch.cuda.synchronize()
            bar.next()

        diff = datetime.now() - previous_time
        train_loss /= self.stats['num_train_batch']
        train_ph_acc /= self.stats['num_train_batch']
        train_st_acc /= self.stats['num_train_batch']

        return train_loss, train_ph_acc, train_st_acc

    def valid_epoch(self):
        valid_loss = 0.0
        valid_ph_acc = 0.0
        valid_st_acc = 0.0

        bar = IncrementalBar('val', max = self.stats['num_valid_batch'])
        for bidx, batch in enumerate(self.valid_loader):
            _, batch_input, batch_phases, batch_steps, video_sup, _ = batch
            batch_input, batch_phases, batch_steps = batch_input.cuda(), batch_phases.cuda(), batch_steps.cuda()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                _, pred_phases, pred_steps, _, _, _ = self.model(batch_input, None)
                loss = self.loss_func(
                                batch_phases, batch_steps,
                                pred_phases, pred_steps, video_sup
                            )
            valid_loss += loss.item()
            if self.add_phases: valid_ph_acc += self.accuracy(batch_phases, pred_phases)
            if self.add_steps: valid_st_acc += self.accuracy(batch_steps, pred_steps)

            bar.next()

        valid_loss /= self.stats['num_valid_batch']
        valid_ph_acc /= self.stats['num_valid_batch']
        valid_st_acc /= self.stats['num_valid_batch']

        return valid_loss, valid_ph_acc, valid_st_acc

    def train(self):
        ## Save training stats after each epoch
        save_dir = self.stats['model_path']
        best_m, best_m_ph_acc, best_m_st_acc = self.stats['best_m_stats']
        start_epoch = self.stats['latest_epoch'] + 1
        logging.info('Starting training from epoch %d/%d' % (start_epoch, self.hp.n_epochs))

        for epoch in range(start_epoch, self.hp.n_epochs):
            train_loss, train_ph_acc, train_st_acc = self.train_epoch()
            valid_loss, valid_ph_acc, valid_st_acc = self.valid_epoch()

            self.stats['experiment_stats'].append((
                epoch, train_loss, valid_loss,
                train_ph_acc, train_st_acc,
                valid_ph_acc, valid_st_acc, time.time()
            ))


            if valid_ph_acc + valid_st_acc >= best_m_ph_acc + best_m_st_acc:
                best_m = 'best'#epoch + 1
                best_m_ph_acc = valid_ph_acc
                best_m_st_acc = valid_st_acc
                self.stats['best_model'] = best_m
                self.stats['best_m_stats'] = [epoch, best_m_ph_acc, best_m_st_acc]
                
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(best_m) + ".model")
                torch.save(self.optimizer.state_dict(), save_dir + "/epoch-" + str(best_m) + ".opt")
                torch.save(self.scaler.state_dict(), save_dir + "/epoch-" + str(best_m) + ".sclr")

                logging.info("[best epoch %d]: epoch loss = %f, val loss = %f, acc ph = %f, acc st = %f,  val acc ph = %f, val acc st = %f" % (
                        self.stats['experiment_stats'][-1][0], self.stats['experiment_stats'][-1][1],
                        self.stats['experiment_stats'][-1][2], self.stats['experiment_stats'][-1][3],
                        self.stats['experiment_stats'][-1][4], self.stats['experiment_stats'][-1][5],
                        self.stats['experiment_stats'][-1][6]
                    )
                )

            logging.info("[epoch %d]: epoch loss = %f, val loss = %f, acc ph = %f, acc st = %f,  val acc ph = %f, val acc st = %f" % (
                    self.stats['experiment_stats'][-1][0], self.stats['experiment_stats'][-1][1],
                    self.stats['experiment_stats'][-1][2], self.stats['experiment_stats'][-1][3],
                    self.stats['experiment_stats'][-1][4], self.stats['experiment_stats'][-1][5],
                    self.stats['experiment_stats'][-1][6]
                )
            )

            # saving latest checkpoint for resume training
            self.stats['latest_epoch'] = epoch
            torch.save(self.model.state_dict(), save_dir + "/epoch-latest" + ".model")
            torch.save(self.optimizer.state_dict(), save_dir + "/epoch-latest" + ".opt")
            torch.save(self.scaler.state_dict(), save_dir + "/epoch-latest" + ".sclr")

            ## Save training stats after each epoch
            with open(os.path.join(save_dir, 'stats.json'), 'w') as fp:
                json.dump(self.stats, fp, ensure_ascii=False, sort_keys=True, indent=4)

        logging.info('Finished training!!!')
        return self.stats

    def test(self, dsets=['train', 'valid', 'test']):
        model_dir = self.stats['model_path']
        epoch = self.stats['best_model']
        logging.info('In test!!!')

        self.model.load_state_dict(torch.load(f'{self.weights_init}/epoch-{str(epoch)}.model'))
        if self.hp.eval_mode:
            self.model.eval()

        with torch.no_grad():
            datasets = [self.train_loader, self.valid_loader, self.test_loader]
            for i, ds in enumerate(dsets):
                output_data = [{
                    'images': [], 'features': [], 'phase_preds': [], 'phase_labels': [],
                    'step_preds': [], 'step_labels': [], 'supervision': False
                }]
                test_loss = 0

                bar = IncrementalBar(ds, max = self.stats['num_' + ds + '_batch'])
                for bidx, batch in enumerate(datasets[i]):
                    batch_images, batch_input, batch_phases, batch_steps, video_sup, video_end = batch
                    batch_input, batch_phases, batch_steps = batch_input.cuda(), batch_phases.cuda(), batch_steps.cuda()
                    with torch.cuda.amp.autocast():
                        features, pred_phases, pred_steps, _, _, _ = self.model(batch_input, None)
                        loss = self.loss_func(
                                        batch_phases, batch_steps,
                                        pred_phases, pred_steps, video_sup
                                    )
                    
                    nz = torch.nonzero(video_end)
                    if nz.numel():
                        nz_idx = nz[0].item()+1
                        output_data[-1]['images'] += batch_images[:nz_idx]
                        output_data[-1]['features'] = np.vstack(output_data[-1]['features'] + [features.detach().cpu().numpy()[:nz_idx]])
                        output_data[-1]['phase_labels'] = np.hstack(output_data[-1]['phase_labels'] + [batch_phases.detach().cpu().numpy()[:nz_idx]])
                        output_data[-1]['step_labels'] = np.hstack(output_data[-1]['step_labels'] + [batch_steps.detach().cpu().numpy()[:nz_idx]])

                        new_video = {
                            'images': [], 'features': [], 'phase_preds': [], 'phase_labels': [],
                            'step_preds': [], 'step_labels': [], 'supervision': False
                        }
                        new_video['images'] += batch_images[nz_idx:]
                        new_video['features'] += [features.detach().cpu().numpy()[nz_idx:]]
                        new_video['phase_labels'] += [batch_phases.detach().cpu().numpy()[nz_idx:]]
                        new_video['step_labels'] += [batch_steps.detach().cpu().numpy()[nz_idx:]]

                        if self.add_phases:
                            _, predicted = torch.max(pred_phases.data, 1)
                            output_data[-1]['phase_preds'] = np.hstack(
                                output_data[-1]['phase_preds'] + \
                                [predicted.detach().cpu().numpy()[:nz_idx]]
                            )
                            new_video['phase_preds'] += [predicted.detach().cpu().numpy()[nz_idx:]]
                        if self.add_steps:
                            _, predicted = torch.max(pred_steps.data, 1)
                            output_data[-1]['step_preds'] = np.hstack(
                                output_data[-1]['step_preds'] + \
                                [predicted.detach().cpu().numpy()[:nz_idx]]
                            )
                            new_video['step_preds'] += [predicted.detach().cpu().numpy()[nz_idx:]]

                        output_data.append(deepcopy(new_video))
                    else:
                        output_data[-1]['images'] += batch_images[:]
                        output_data[-1]['features'] += [features.detach().cpu().numpy()]
                        output_data[-1]['phase_labels'] += [batch_phases.detach().cpu().numpy()]
                        output_data[-1]['step_labels'] += [batch_steps.detach().cpu().numpy()]
                        output_data[-1]['supervision'] = bool(video_sup[0])
                        if self.add_phases:
                            _, predicted = torch.max(pred_phases.data, 1)
                            output_data[-1]['phase_preds'] += [predicted.detach().cpu().numpy()]

                        if self.add_steps:
                            _, predicted = torch.max(pred_steps.data, 1)
                            output_data[-1]['step_preds'] += [predicted.detach().cpu().numpy()]

                    test_loss += loss.item()
                    bar.next()

                test_loss /= len(datasets[i])
                logging.info('%s loss: %f' %(ds, test_loss))

                self.stats[ds + '_loss'] = test_loss
                
                with open(os.path.join(model_dir, ds + '_imgs.npy'), 'wb') as fp:
                    np.save(fp, np.concatenate([d['images'] for d in output_data[:-1]]))

                if self.add_phases:
                    with open(os.path.join(model_dir, ds + '_phase_labels.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate([d['phase_labels'] for d in output_data[:-1]]))
                    with open(os.path.join(model_dir, ds + '_phase_preds.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate([d['phase_preds'] for d in output_data[:-1]]))
                if self.add_steps:
                    with open(os.path.join(model_dir, ds + '_step_labels.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate([d['step_labels'] for d in output_data[:-1]]))
                    with open(os.path.join(model_dir, ds + '_step_preds.npy'), 'wb') as fp:
                        np.save(fp, np.concatenate([d['step_preds'] for d in output_data[:-1]]))
                
                ## Save pkl file with feature embeddings
                with open(os.path.join(model_dir, f'{ds}_videos.pickle'), 'wb') as fp:
                    pickle.dump(output_data[:-1], fp, pickle.HIGHEST_PROTOCOL)

                ## Save training stats after each epoch
                with open(os.path.join(model_dir, 'stats_test.json'), 'w') as fp:
                    json.dump(self.stats, fp, ensure_ascii=False, sort_keys=True, indent=4)
        return
