'''
This file is only used in the ipynb file.
'''

import os
import sys
import torch

class Hparams:
    args = {
        'save_model_dir': './results/lr1e-3',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dataset_root': './data_mini/',
        'sampling_rate': 16000,
        'sample_length': 5,  # in second
        'num_workers': 0,  # Number of additional thread for data loading. A large number may freeze your laptop.
        'annotation_path': './data_mini/annotations.json',

        'frame_size': 0.02,
        'batch_size': 8,  # 32 produce best result so far
    }
