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
--params_dict_path experiments/2024-07-10/optimize_labeling/process_files_batch/1_smooth_drums.json \
--export_path experiments/2024-07-10/flatten_list/process_files_from_params/single/vocals_optimize_labeling_1_smooth_drums.wav
"""

def apply_effects_to_sig(audio_dir_or_file: Union[str, Path, List[str], List[Path]], params_dict_path: Union[str, Path], export_path: str):
    # audio_dir_or_files = Path(audio_dir_or_files)

    if isinstance(audio_dir_or_file, (str, Path)):
        audio_dir_or_file = Path(audio_dir_or_file)
        if audio_dir_or_file.is_dir():
            print(f'{audio_dir_or_file} is directory')
            in_sig = tc.wavs_to_batch(audio_dir_or_file)
        else:
            print(f'{audio_dir_or_file} is a file')
            in_sig = tc.preprocess_audio(audio_dir_or_file).to(DEVICE)
    else:
        print(f'{audio_dir_or_file} is a list of files')
        in_sig = tc.wavs_to_batch(audio_dir_or_file)


    with open(params_dict_path, 'r') as f:
        params_dict = json.load(f)

    fx_chain = list(params_dict.keys())
    fx_channel = tc.create_channel(fx_chain)

    params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
    if in_sig.batch_size != 1:
        params_list = params_list.transpose(0, 1)
    breakpoint()
    params = params_list.expand(in_sig.batch_size, -1).to(DEVICE)

    out_sig = fx_channel(in_sig.clone().to(DEVICE), torch.sigmoid(params))

    tc.export_sig(out_sig.clone().detach().cpu(), export_path)
    
    return out_sig


# # testing single file
# apply_effects_to_sig(
#     audio_dir_or_file='assets/multistem_examples/10s/drums.wav',
#     params_dict_path='experiments/2024-07-10/optimize_labeling/process_files_batch/2_tinny_vocals.json',
#     export_path='experiments/2024-07-10/applyFXfromparams/single_2.wav'
# )

# testing multiple files
apply_effects_to_sig(
    audio_dir_or_file=['assets/multistem_examples/10s/drums.wav']*3,
    params_dict_path='experiments/2024-07-10/optimize_labeling/process_files_batch/output_all.json',
    export_path='experiments/2024-07-10/applyFXfromparams/wav_dir_multi_json2'
)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Apply effects to an audio file or dir of audio files based on parameters in a JSON file and export the processed file.")
    
#     parser.add_argument('--audio_dir_or_file', type=str, required=True, help="Path to the input audio file.")
#     parser.add_argument('--params_dict_path', type=str, required=True, help="Path to the JSON file containing effect parameters.")
#     parser.add_argument('--export_path', type=str, required=True, help="Path to save the processed audio file.")

#     args = parser.parse_args()

#     apply_effects_to_sig(
#         audio_dir_or_file=args.audio_dir_or_file,
#         params_dict_path=args.params_dict_path,
#         export_path=args.export_path
#     )

