from pathlib import Path
from typing import Union, List, Optional

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.core import Channel
from text2fx.__main__ import text2fx
from text2fx.constants import DEVICE, SAMPLE_RATE
import torch
import json

"""
Applies effects to an audio file based on parameters from a JSON dictionary.

:param audio_path: Path to the input audio file.
:param params_dict_path: Path to the JSON dictionary file containing de-normalized effect parameters.
:param output_path: Path to save the processed audio file.
"""

def apply_effects(audio_file: Union[str, Path], params_dict_path: Union[str, Path], export_path: str):
    in_sig = tc.preprocess_audio(audio_file).to(DEVICE)
    with open(params_dict_path, 'r') as f:
        params_dict = json.load(f)

    fx_chain = list(params_dict.keys())
    fx_channel = tc.create_channel(fx_chain)

    params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
    params = params_list.expand(in_sig.batch_size, -1).to(DEVICE)

    out_sig = fx_channel(in_sig.clone(), torch.sigmoid(params))

    tc.export_sig(out_sig, export_path)
    

apply_effects(
    audio_file='assets/multistem_examples/10s/vocals.wav',
    params_dict_path='experiments/2024-07-08/process_FILES_test/output_0_bass_happy.json',
    export_path='experiments/2024-07-08/process_from_params/output_single.wav'
)