from pathlib import Path
from typing import Union, List, Optional

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.core import Channel
from text2fx.__main__ import text2fx
from text2fx.constants import DEVICE, SAMPLE_RATE
import torch
import json
import argparse

"""
Applies effects to an audio file or directory of audio files based on parameters from a JSON dictionary.

:param audio_path: Path to the input audio file or directory
:param params_dict_path: Path to the JSON dictionary file containing de-normalized effect parameters.
:param output_path: Path to save the processed audio file.


python process_file_from_params.py --audio_dir_or_file assets/multistem_examples/10s/vocals.wav \
--params_dict_path experiments/2024-07-09/checking_process_files/output_4_drums_cold.json \
--export_path experiments/2024-07-09/checking_process_file_from_params_2/output_single_vocals_list.wav
"""

def apply_effects_to_sig(audio_dir_or_file: Union[str, Path], params_dict_path: Union[str, Path], export_path: str):
    audio_dir_or_file = Path(audio_dir_or_file)

    if audio_dir_or_file.is_dir():
        in_sig = tc.wav_dir_to_batch(audio_dir_or_file).to(DEVICE)
    else:
        in_sig = tc.preprocess_audio(audio_dir_or_file).to(DEVICE)

    with open(params_dict_path, 'r') as f:
        params_dict = json.load(f)

    # flattening so "low_shelf_gain_db": [scalar] ==> "low_shelf_gain_db": scalar
    params_dict = {key: {inner_key: inner_value[0] for inner_key, inner_value in value.items()} for key, value in params_dict.items()}

    fx_chain = list(params_dict.keys())
    fx_channel = tc.create_channel(fx_chain)

    params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
    breakpoint()
    params = params_list.expand(in_sig.batch_size, -1).to(DEVICE)

    out_sig = fx_channel(in_sig.clone(), torch.sigmoid(params))

    tc.export_sig(out_sig, export_path)
    
    return out_sig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply effects to an audio file or dir of audio files based on parameters in a JSON file and export the processed file.")
    
    parser.add_argument('--audio_dir_or_file', type=str, required=True, help="Path to the input audio file.")
    parser.add_argument('--params_dict_path', type=str, required=True, help="Path to the JSON file containing effect parameters.")
    parser.add_argument('--export_path', type=str, required=True, help="Path to save the processed audio file.")

    args = parser.parse_args()

    apply_effects_to_sig(
        audio_dir_or_file=args.audio_dir_or_file,
        params_dict_path=args.params_dict_path,
        export_path=args.export_path
    )

