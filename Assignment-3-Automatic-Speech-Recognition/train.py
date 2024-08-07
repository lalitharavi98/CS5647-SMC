#!/usr/bin/env/python3
"""This minimal example trains a CTC-based speech recognizer on a tiny dataset.
The encoder is based on a combination of convolutional, recurrent, and
feed-forward networks (CRDNN) that predict phonemes.  A greedy search is used on
top of the output probabilities.
Given the tiny dataset, the expected behavior is to overfit the training dataset
(with a validation performance that stays high).
"""
import sys
import pathlib

import speechbrain
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from utils import *
import torch
from speechbrain.utils.checkpoints import Checkpointer


def main(device="cpu"):
    experiment_dir = pathlib.Path(__file__).resolve().parent
    hparam_fn = sys.argv[1]
    hparams_file = experiment_dir / hparam_fn
    data_folder = './datasets/tiny_librispeech'
    data_folder = (experiment_dir / data_folder).resolve()

    # Load model hyper parameters:
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)

    # Dataset creation
    train_data, valid_data = data_prep(data_folder, hparams)

    # Trainer initialization
    ctc_brain = CTCBrain(
        hparams["modules"],
        hparams["opt_class"],
        hparams,
        run_opts={"device": device},
    )

    # Training/validation loop
    ctc_brain.fit(
        range(hparams["N_epochs"]),
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Evaluation is run separately (now just evaluating on valid data)
    ctc_brain.evaluate(valid_data, min_key='PER')


def data_prep(data_folder, hparams):
    "Creates the datasets and their data processing pipelines."

    annot_dir = jpath(data_folder, 'annotation')
    tokenizer = PhonemeTokenizer()

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / jpath(annot_dir, "train.json"),
        replacements={"data_root": data_folder},
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=data_folder / jpath(annot_dir, "valid.json"),
        replacements={"data_root": data_folder},
    )
    datasets = [train_data, valid_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phn")
    @sb.utils.data_pipeline.provides("phn_list", "phn_encoded")
    def text_pipeline(phn):
        phn_list = phn.strip().split()
        phn_list = [a.upper() for a in phn_list]
        yield phn_list
        phn_encoded = tokenizer.encode_seq(phn_list)
        phn_encoded = torch.tensor(phn_encoded, dtype=torch.long)
        yield phn_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "phn_encoded"])

    return train_data, valid_data


class CTCBrain(sb.Brain):
    def on_fit_start(self):
        super().on_fit_start()  # resume ckpt
        self.tokenizer = PhonemeTokenizer()

    def compute_forward(self, batch, stage):
        "Given an input batch it computes the output probabilities."
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        x = self.modules.model(feats)
        x = self.modules.lin(x)
        outputs = self.hparams.softmax(x)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        "Given the network predictions and targets computed the CTC loss."
        predictions, lens = predictions
        phns, phn_lens = batch.phn_encoded
        loss = self.hparams.compute_cost(predictions, phns, lens, phn_lens)

        if stage != sb.Stage.TRAIN:
            seq = sb.decoders.ctc_greedy_decode(
                predictions, lens, blank_id=self.hparams.blank_index
            )
            t = phns.tolist()
            out = self.tokenizer.decode_seq_batch(seq)
            tgt = self.tokenizer.decode_seq_batch(t)
            self.per_metrics.append(batch.id, out, tgt)

        return loss

    def on_stage_start(self, stage, epoch=None):
        "Gets called when a stage (either training, validation, test) starts."
        if stage != sb.Stage.TRAIN:
            self.per_metrics = self.hparams.per_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of a stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        elif stage == sb.Stage.VALID:
            PER = self.per_metrics.summarize("error_rate")
            print("Epoch %d complete" % epoch)
            print("Train loss: %.2f" % self.train_loss)
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % PER)

        elif stage == sb.Stage.TEST:
            PER = self.per_metrics.summarize("error_rate")
            print(stage, "loss: %.2f" % stage_loss)
            print(stage, "PER: %.2f" % PER)


if __name__ == "__main__":
    main()
