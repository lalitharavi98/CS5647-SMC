import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from model import BaseCNN_mini
from dataset import get_data_loader, move_data_to_device
from utils import ls
# import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')


# os.environ["CUDA_VISIBLE_DEVICES"] = '3' # If you have multiple GPU's,
# uncomment this line to specify which GPU you want to use


def main():
    args = {
        'save_model_dir': './results',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_root': './data_mini/',
        'sampling_rate': 16000,
        'sample_length': 5,  # in second
        'num_workers': 0,  # Number of additional thread for data loading. A large number may freeze your laptop.

        'frame_size': 0.02,
        'batch_size': 8,  # 32 produce best result so far
    }

    ast_model = AST_Model(args['device'])

    # Set learning params
    learning_params = {
        'batch_size': 50,
        'epoch': 10,
        'lr': 1e-4,
    }

    # Train and Validation
    best_model_id = ast_model.fit(args, learning_params)
    print("Best model from epoch: ", best_model_id)


class AST_Model:
    '''
    This is main class for training model and making predictions.
    '''

    def __init__(self, device="cpu", model_path=None):
        # Initialize model
        self.device = device
        self.model = BaseCNN_mini(feat_dim=256).to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded.')
        else:
            print('Model initialized.')

    def fit(self, args, learning_params):
        # Set paths
        save_model_dir = args['save_model_dir']
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_params['lr'])
        loss_func = LossFunc(device=self.device)
        metric = Metrics(loss_func)

        train_loader = get_data_loader(split='train', args=args)
        valid_loader = get_data_loader(split='valid', args=args)

        # Start training
        print('Start training...')
        start_time = time.time()
        best_model_id = -1
        min_valid_loss = 10000

        for epoch in range(1, learning_params['epoch'] + 1):
            self.model.train()
            total_training_loss = 0

            # Train
            pbar = tqdm(train_loader)
            for batch_idx, batch in enumerate(pbar):
                x, onset, offset, octave, pitch_class = move_data_to_device(batch, args['device'])
                tgt = onset, offset, octave, pitch_class
                out = self.model(x)
                losses = loss_func.get_loss(out, tgt)
                loss = losses[0]
                metric.update(out, tgt, losses)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()

                pbar.set_description('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
            metric_train = metric.get_value()

            # Validation
            self.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_loader):
                    x, onset, offset, octave, pitch_class = move_data_to_device(batch, args['device'])
                    tgt = onset, offset, octave, pitch_class
                    out = self.model(x)
                    metric.update(out, tgt)
            metric_valid = metric.get_value()

            # Logging
            print('[Epoch {:02d}], Train Loss: {:.5f}, Valid Loss {:.5f}, Time {:.2f}s'.format(
                epoch, metric_train['loss'], metric_valid['loss'], time.time() - start_time,
            ))
            print('Split Train F1/Accuracy: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_train['onset_f1'],
                metric_train['offset_f1'],
                metric_train['octave_acc'],
                metric_train['pitch_acc']
            ))
            print('Split Valid F1/Accuracy: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_valid['onset_f1'],
                metric_valid['offset_f1'],
                metric_valid['octave_acc'],
                metric_valid['pitch_acc']
            ))
            print('Split Train Loss: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_train['onset_loss'],
                metric_train['offset_loss'],
                metric_train['octave_loss'],
                metric_train['pitch_loss']
            ))
            print('Split Valid Loss: Onset {:.4f} | Offset {:.4f} | Octave {:.4f} | Pitch Class {:.4f}'.format(
                metric_valid['onset_loss'],
                metric_valid['offset_loss'],
                metric_valid['octave_loss'],
                metric_valid['pitch_loss']
            ))

            # Save the best model
            if metric_valid['loss'] < min_valid_loss:
                min_valid_loss = metric_valid['loss']
                best_model_id = epoch

                save_dict = self.model.state_dict()
                target_model_path = save_model_dir + '/best_model.pth'
                torch.save(save_dict, target_model_path)

        print('Training done in {:.1f} minutes.'.format((time.time() - start_time) / 60))
        return best_model_id

    def parse_frame_info(self, frame_info, args):
        """
        Convert frame-level output into note-level predictions.
        """

        frame_num = len(frame_info)

        result = []
        current_onset = None
        pitch_counter = []
        local_max_size = 3
        current_frame = 0.0

        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
        onset_seq_length = len(onset_seq)

        frame_length = args['frame_size']

        for i in range(frame_num):  # For each frame
            current_frame = frame_length * i
            info = frame_info[i]
            last_frame = max(0, current_frame - 1)

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # If the frame is an onset
            if info[0]:
                if current_onset is None:
                    current_onset = current_frame
                else:
                    if len(pitch_counter) > 0:
                        pitch = max(set(pitch_counter), key=pitch_counter.count) + 36
                        result.append([current_onset, current_frame, pitch])
                    current_onset = current_frame
                    pitch_counter = []

            # If it is offset
            elif info[1]:
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        pitch = max(set(pitch_counter), key=pitch_counter.count) + 36
                        result.append([current_onset, current_frame, pitch])
                    current_onset = None
                    pitch_counter = []
                else:
                    pass

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                if info[2] != 0 and info[3] != 0:
                    current_pitch = int((info[2] - 1) * 12 + (info[3] - 1))
                    pitch_counter.append(current_pitch)

        # The last note
        if current_onset is not None:
            if len(pitch_counter) > 0:
                pitch = max(set(pitch_counter), key=pitch_counter.count) + 36
                result.append([current_onset, current_frame, pitch])

        return result

    def predict(self, testset_path, onset_thres, offset_thres, args):
        """Predict results for a given test dataset."""
        songs = ls(testset_path)
        results = {}
        for song in songs:
            if song.startswith('.'):
                continue
            test_loader = get_data_loader(split='test', fns=[song], args=args)

            # Start predicting
            self.model.eval()
            with torch.no_grad():
                on_frame = []
                off_frame = []
                oct_frame = []
                pitch_frame = []
                loss_func = LossFunc(args['device'])
                metric = Metrics(loss_func)
                pbar = tqdm(test_loader)
                for batch_idx, batch in enumerate(pbar):
                    x, onset, offset, octave, pitch_class = move_data_to_device(batch, self.device)
                    tgt = onset, offset, octave, pitch_class
                    out = self.model(x)
                    metric.update(out, tgt)

                    # Collect frames for corresponding songs
                    on_out = out[0].flatten()
                    on_out[on_out >= onset_thres] = 1
                    on_out[on_out < onset_thres] = 0
                    on_out = on_out.long()
                    off_out = out[1].flatten()
                    off_out[off_out >= offset_thres] = 1
                    off_out[off_out < offset_thres] = 0
                    off_out = off_out.long()
                    oct_out = torch.argmax(out[2], dim=2).flatten()
                    pitch_out = torch.argmax(out[3], dim=2).flatten()

                    on_frame.append(on_out)
                    off_frame.append(off_out)
                    oct_frame.append(oct_out)
                    pitch_frame.append(pitch_out)

                on_out = torch.cat(on_frame).tolist()
                off_out = torch.cat(off_frame).tolist()
                oct_out = torch.cat(oct_frame).tolist()
                pitch_out = torch.cat(pitch_frame).tolist()
                frame_info = list(zip(on_out, off_out, oct_out, pitch_out))

                # Parse frame info into output format for every song
                results[song] = self.parse_frame_info(frame_info=frame_info, args=args)

        return results


class LossFunc:
    def __init__(self, device):
        self.device = device
        '''
        We will use Binary Cross Entropy for onset and offset classification, with 15 as weight for positive value
        Cross Entropy Loss for octave and pitch class classification.
        
        NOTE: raw model output will not be normalized by softmax/sigmoid function. 
        
        YOUR CODE: finish the __init__ and get_loss function.
        '''
        self.onset_criterion = None
        self.offset_criterion = None
        self.octave_criterion = None
        self.pitch_criterion = None

    def get_loss(self, out, tgt):
        '''
        This function receive model output and target for onset, offset, octave, and pitch class, then
        compute loss for the 4 pairs respectively,
        finally add them together (simple addition, no weight) as the total loss
        Return: (total loss, onset loss, offset loss, octave loss, pitch loss)
        '''

        '''
        YOUR CODE HERE: finish the function
        '''

        return None


class Metrics:
    def __init__(self, loss_func):
        self.buffer = {}
        self.loss_func = loss_func

    def update(self, out, tgt, losses=None):
        '''
        Compute metrics for one batch of output and target.
        F1 score for onset and offset,
        Accuracy for octave and pitch class.
        Append the results to a list, and link the list to self.buffer[metric_name].
        '''
        with torch.no_grad():
            out_on, out_off, out_oct, out_pitch = out
            tgt_on, tgt_off, tgt_oct, tgt_pitch = tgt

            if losses == None:
                losses = self.loss_func.get_loss(out, tgt)

            '''
            YOUR CODE HERE: compute the four metrics below.
            '''
            onset_f1 = None
            offset_f1 = None
            oct_acc = None
            pitch_acc = None

            batch_metric = {
                'loss': losses[0].item(),
                'onset_loss': losses[1].item(),
                'offset_loss': losses[2].item(),
                'octave_loss': losses[3].item(),
                'pitch_loss': losses[4].item(),
                'onset_f1': onset_f1,
                'offset_f1': offset_f1,
                'octave_acc': oct_acc,
                'pitch_acc': pitch_acc,
            }

            for k in batch_metric:
                if k in self.buffer:
                    self.buffer[k].append(batch_metric[k])
                else:
                    self.buffer[k] = [batch_metric[k]]

    def get_value(self):
        for k in self.buffer:
            self.buffer[k] = sum(self.buffer[k]) / len(self.buffer[k])
        ret = self.buffer
        self.buffer = {}
        return ret


if __name__ == '__main__':
    main()
