import math
import json
import torch
import librosa
import torchaudio
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from utils import read_json, save_json, ls, jpath


def move_data_to_device(data, device):
    ret = []
    for i in data:
        if isinstance(i, torch.Tensor):
            ret.append(i.to(device))
    return ret


def get_data_loader(split, args, fns=None):
    dataset = MyDataset(
        dataset_root=args['dataset_root'],
        split=split,
        sampling_rate=args['sampling_rate'],
        annotation_path=args['annotation_path'],
        sample_length=args['sample_length'],
        frame_size=args['frame_size'],
        song_fns=fns,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def collate_fn(batch):
    '''
    This function help to
    1. Group different components into separate tensors.
    2. Pad samples in the maximum length in the batch.
    '''

    inp = []
    onset = []
    offset = []
    octave = []
    pitch = []
    max_frame_num = 0
    for sample in batch:
        max_frame_num = max(max_frame_num, sample[0].shape[0], sample[1].shape[0], sample[2].shape[0],
                            sample[3].shape[0], sample[4].shape[0])

    for sample in batch:
        inp.append(
            torch.nn.functional.pad(sample[0], (0, 0, 0, max_frame_num - sample[0].shape[0]), mode='constant', value=0))
        onset.append(
            torch.nn.functional.pad(sample[1], (0, max_frame_num - sample[1].shape[0]), mode='constant', value=0))
        offset.append(
            torch.nn.functional.pad(sample[2], (0, max_frame_num - sample[2].shape[0]), mode='constant', value=0))
        octave.append(
            torch.nn.functional.pad(sample[3], (0, max_frame_num - sample[3].shape[0]), mode='constant', value=0))
        pitch.append(
            torch.nn.functional.pad(sample[4], (0, max_frame_num - sample[4].shape[0]), mode='constant', value=0))

    inp = torch.stack(inp)
    onset = torch.stack(onset)
    offset = torch.stack(offset)
    octave = torch.stack(octave)
    pitch = torch.stack(pitch)

    return inp, onset, offset, octave, pitch


class MyDataset(Dataset):
    def __init__(self, dataset_root, split, sampling_rate, annotation_path, sample_length, frame_size,
                 song_fns=None):
        '''
        This dataset return an audio clip in a specific duration in the training loop, with its "__getitem__" function.
        '''
        self.dataset_root = dataset_root
        self.split = split
        self.dataset_path = jpath(self.dataset_root, self.split)
        self.sampling_rate = sampling_rate
        self.annotation_path = annotation_path
        self.all_annotations = read_json(self.annotation_path)
        self.duration = {}
        if song_fns == None:
            self.song_fns = ls(self.dataset_path)
            self.song_fns.sort()
        else:
            self.song_fns = song_fns
        self.index = self.index_data(sample_length)
        self.sample_length = sample_length
        self.frame_size = frame_size
        self.frame_per_sec = int(1 / self.frame_size)

    def index_data(self, sample_length):
        '''
        Prepare the index for the dataset, i.e., the audio file name and starting time of each sample
        '''
        index = []
        for song_fn in self.song_fns:
            if song_fn.startswith('.'):  # Ignore any hidden file
                continue
            duration = self.all_annotations[song_fn][-1][1]
            num_seg = math.ceil(duration / sample_length)
            for i in range(num_seg):
                index.append([song_fn, i * sample_length])
            self.duration[song_fn] = duration
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        '''
        Return spectrogram and 4 labels of an audio clip
        The audio filename and the start time of this sample is specified by "audio_fn" and "start_sec"
        '''
        audio_fn, start_sec = self.index[idx]
        end_sec = start_sec + self.sample_length

        audio_fp = jpath(self.dataset_path, audio_fn, 'Mixture.mp3')

        ''' YOUR CODE: Load audio from file, and compute mel spectrogram '''
        mel_spectrogram = None

        duration = self.duration[audio_fn]
        onset_roll, offset_roll, octave_roll, pitch_roll = self.get_labels(self.all_annotations[audio_fn], duration)

        ''' YOUR CODE: Extract the desired clip, i.e., 5 sec of info, from both spectrogram and annotation '''
        ''' The clip start from start_sec, end at end_sec '''
        spectrogram_clip = None
        onset_clip = None
        offset_clip = None
        octave_clip = None
        pitch_class_clip = None

        return spectrogram_clip, onset_clip, offset_clip, octave_clip, pitch_class_clip

    def get_labels(self, annotation_data, duration):
        '''
        This function read annotation from file, and then convert annotation from note-level to frame-level
        Because we will be using frame-level labels in training.
        '''
        frame_num = math.ceil(duration * self.frame_per_sec)

        octave_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
        pitch_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
        onset_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
        offset_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)

        ''' YOUR CODE: Create frame level label for a song to facilitate consequent computation '''
        ''' They are: onset roll, offset roll, octave roll, and pitch class roll '''
        ''' Each XX roll is a vector with integer elements, vector length equals to the number of frames of the song '''
        ''' Value range for onset, offset, octave, pitch class: [0,1], [0,1], [0,4], [0,12] '''
        ''' For onset and offset, 1 means there exists onset/offset in this frame '''
        ''' For octave and pitch class, 0 means illegal pitch or silence '''

        return onset_roll, offset_roll, octave_roll, pitch_roll

    def get_octave_and_pitch_class_from_pitch(self, pitch, note_start=36):
        '''
        Convert MIDI pitch number to octave and pitch_class
        pitch: int, range [36 (octave 0, pitch_class 0), 83 (octave 3, pitch 11)]
                pitch = 0 means silence
        return: octave, pitch_class.
                if no pitch or pitch out of range, output: 0, 0
        '''
        # note numbers ranging from C2 (36) to B5 (83)
        # octave class ranging from 0 to 4, 1~4 are valid octave class, 0 represent silence
        # pitch_class ranging from 0 to 12, pitch class 1 to 12: pitch C to B, pitch class 0: unknown class / silence
        if pitch == 0:
            return 4, 12

        t = pitch - note_start
        octave = t // 12
        pitch_class = t % 12

        if pitch < note_start or pitch > 83:
            return 0, 0
        else:
            return octave + 1, pitch_class + 1
