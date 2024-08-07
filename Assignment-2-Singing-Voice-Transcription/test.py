import torch
from torch.utils.data import DataLoader

import os
import json
import argparse
import mido
from tqdm import tqdm
from pathlib import Path
from train import AST_Model, Metrics, LossFunc
from evaluate import MirEval
from utils import ls, jpath, save_json

import warnings

warnings.filterwarnings('ignore')


def main():
    args = {
        'dataset_root': './data_mini/',
        'test_dataset_path': './data_mini/test',
        'output_dir': './results',
        'output_path': './results/predictions.json',
        'save_model_dir': './results',
        'sampling_rate': 16000,
        'sample_length': 5,  # in second
        'num_workers': 0,
        'onset_thres': 0.7,
        'offset_thres': 0.5,

        'batch_size': 1,

        'annotation_path': './data/annotations.json',
        'predicted_json_path': './results/predictions.json',
        'tolerance': 0.1,

        'frame_size': 0.02,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_model_path = os.path.join(args['save_model_dir'], 'best_model.pth')
    model = AST_Model(device, best_model_path)

    results = model.predict(testset_path=args['test_dataset_path'], onset_thres=args['onset_thres'],
                            offset_thres=args['offset_thres'], args=args)
    save_json(results, args['output_path'], sort=True)

    # convert_to_midi(results, '91', jpath(args['output_dir'], 'out.mid'))

    my_eval = MirEval()
    my_eval.prepare_data(args['annotation_path'], args['predicted_json_path'])
    my_eval.accuracy(onset_tolerance=float(args['tolerance']))


def notes2mid(notes):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))

        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1] - 0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid


def convert_to_midi(predicted_result, song_id, output_path):
    to_convert = predicted_result[song_id]
    mid = notes2mid(to_convert)
    mid.save(output_path)


if __name__ == '__main__':
    """
    This script performs inference using the trained singing transcription model in main.py.
    
    Sample usage:
    python inference.py --best_model_id 9 
    The best model may not be number 9. It depends on your result of validation.
    """
    main()
