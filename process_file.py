import os
import shutil
import audiotools as at
from audiotools import AudioSignal
from pathlib import Path
import dasp_pytorch
from typing import Union, List, Optional, Tuple, Iterable, Dict


import torch
import text2fx.core as tc
from text2fx.core import ParametricEQ_40band, Channel, functional_parametric_eq_40band
from text2fx.constants import EQ_freq_bands, EQ_words_top_10, NOTEBOOKS_DIR, SAMPLE_RATE, DEVICE, EQ_GAINS_PATH
from text2fx.__main__ import text2fx, get_default_channel

"""
A script process_file.py that processes a single audio file 
(given as a path) with a given FX chain (given as a list of strings)
 to match a description (given as a string);
   in addition to taking optional arguments (learning rate, number of steps, loss type, parameter initialization, augmentation params)
   , the script should save 
        - a dictionary of optimized effect controls as json (to a save path given as an argument) 
        - optionally an optimized audio file (saved to a given audio save path).
"""
def single_file_to_sig_batch(file_path, word_list):
    pass

def optimize():
    # return a dict of of optimized paramters
    pass


# def apply_effects(audio, sr, fx_chain, controls):
#     # Placeholder for applying effects logic
#     processed_audio = audio  # Replace with actual audio processing logic
#     return processed_audio

def singletest(in_sig, text_target, channel, save_dir=None):
    signal_effected, sig_effected_params = text2fx(
            model_name='ms_clap', 
            sig=in_sig, 
            text=text_target, 
            channel=channel,
            criterion='cosine-sim', 
            # save_dir=save_dir / text_target / f'paramdict',
            params_init_type='random',
            n_iters=50,
            # roll_amt=0,
            # export_audio=False,
            log_tensorboard=True
    )

def main(audio_path: Union[str, Path, AudioSignal], 
         fx_chain: List[str], 
         text_target: str, 
         export_param_dict_path: str = None, 
         export_audio_path: str = None):
    
    in_sig = tc.preprocess_audio(audio_path)

    fx_channel = tc.create_channel(fx_chain)

    signal_effected, sig_effected_params = text2fx(
        model_name='ms_clap', 
        sig=in_sig, 
        text=text_target, 
        channel=fx_channel,
        criterion='cosine-sim', 
        params_init_type='random',
        n_iters=50,
        # roll_amt=0,
        # export_audio=False,
        log_tensorboard=False
    )

    out_params_dict = fx_channel.save_params_to_dict(sig_effected_params)
    breakpoint()
    if export_audio_path is not None:
        tc.export_sig(signal_effected, export_audio_path)
    
    if export_param_dict_path is not None:
        tc.save_dict_to_json(out_params_dict,export_param_dict_path)

    return out_params_dict

main(
    audio_path='assets/multistem_examples/10s/guitar.wav',
    fx_chain=['EQ', 'reverb'],
    text_target='warm',
    export_param_dict_path= f'/home/annie/research/text2fx/experiments/2024-07-08/process_file_test/output.json',
    export_audio_path = f'/home/annie/research/text2fx/experiments/2024-07-08/process_file_test/final_audio.wav'
)

