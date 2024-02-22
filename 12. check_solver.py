import logging
from pathlib import Path
import os
import sys
import mne
import torch
import numpy as np
import bm
from bm import play
from bm.train import main
from bm.events import Word
from matplotlib import pyplot as plt
from IPython import display as disp

mne.set_log_level(False)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
os.chdir(main.dora.dir.parent)
os.environ['NO_DOWNLOAD'] = '1'

def _get_segments_and_vocabs(solver):
    from scripts.run_eval_probs import _get_extra_info
    print(solver.args.num_workers)
    per_split = {}
    for split in ['train','test','val']:
        segments,vocab, estimates,outputs,features_masks, reject_masks = [], [],[], [],[], []
        dset = getattr(solver.datasets, split)
        loader = solver.make_loader(dset, shuffle=False)
        test_features = solver.datasets.test.datasets[0].features
        for idx, batch in enumerate(loader):
            with torch.no_grad():
                if split =="test":
                    features = test_features.extract_features(batch.features, solver.used_features.keys())
                    estimate, output, features_mask, reject_mask = solver._process_batch(batch.replace(features=features))
                else: 
                     estimate, output, features_mask, reject_mask = solver._process_batch(batch)
                data,  words, word_segs= _get_extra_info(batch, solver.args.dset.sample_rate)
                segments.append(word_segs)
                vocab.append(words)
                estimates.append(estimate.detach().cpu())
                outputs.append(output.detach().cpu())                
        estimates = torch.cat(estimates, dim=0)
        outputs = torch.cat(outputs, dim=0)
        per_split[split] = (segments, vocab, estimates,outputs)
    return per_split



if __name__ == "__main__":
    sig='04405661'
    solver = play.get_solver_from_sig(sig) 
    per_split =_get_segments_and_vocabs(solver)
    new_segments_test=[]
    for segment in segments:
        for elem in segment:
            new_segments_test.append(elem)
    segments_train, vocab_train, _,_ =per_split['test']
    new_segments_train=[]
    for segment in segments_train:
        for elem in segment:
            new_segments_train.append(elem)
    intersetction = set(new_segments_train) & set(new_segments_test) 
    print(intersetction)
    with open('your_file.txt', 'w') as f:
        for line in list(intersetction):
            f.write(f"{line}\n")