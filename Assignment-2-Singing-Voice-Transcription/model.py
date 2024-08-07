import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN_mini(nn.Module):
    '''
    This is a base CNN model.
    '''

    def __init__(self, feat_dim=256, pitch_class=13, pitch_octave=5):
        '''
        Definition of network structure.
        '''
        super().__init__()
        self.feat_dim = 256
        self.pitch_octave = pitch_octave
        self.pitch_class = pitch_class

        '''
        YOUR CODE: the remaining part of the model structure
        '''

    def forward(self, x):
        '''
        Compute output from input
        '''
        '''
        YOUR CODE: computing output from input
        '''

        onset_logits = None
        offset_logits = None
        pitch_octave_logits = None
        pitch_class_logits = None

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits
