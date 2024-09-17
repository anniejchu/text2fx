from pathlib import Path
from typing import Union, List, Optional

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.core import Channel
# from text2fx.__main__ import text2fx
import text2fx

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

def normalize_param_dict(param_dict: dict, channel) -> dict:
    """Given parameters on (0,1) restore them to the ranges expected by the processor.

    Args:
        param_dict (dict): Dictionary of parameter tensors on (0,1) for each module.
        modules (list): List of modules corresponding to the keys in param_dict.

    Returns:
        dict: Dictionary of parameter tensors on their full range for all modules.
    """
    def norm(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)
    all_denorm_params = {}
    modules = channel.modules
    print(modules)
    for fx_name, fx_params in param_dict.items():
        denorm_param_dict = {}

        # Find the corresponding module based on the class name (fx_name)
        fx = next((m for m in modules if m.__class__.__name__ == fx_name), None)
        
        if fx is None:
            raise ValueError(f"No module found with name {fx_name}")

        # Loop through each parameter in the module (fx)
        for param_name, param_tensor in fx_params.items():
            # Denormalize the parameter using its range from fx.param_ranges
            param_val_denorm = norm(
                param_tensor,#[0],  # Assuming param_tensor is a tensor and taking the first element
                fx.param_ranges[param_name][0],
                fx.param_ranges[param_name][1],
            )
            denorm_param_dict[param_name] = param_val_denorm

        # Store the denormalized parameters for this module
        all_denorm_params[fx_name] = denorm_param_dict

    return all_denorm_params


def apply_fx_to_sig(
        audio_dir_or_file: Union[str, Path, List[str], List[Path]], 
        params_dict_path: Union[str, Path],
        export_path: str):
    """
    Assumes param_dict json is formatted like 
        {
            'Effect1': {'param1': value1, 'param2': value2, ...},
            'Effect2': {'param1': value3, 'param2': value4, ...},
            ...
        }
"""
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
        _params_dict = json.load(f)

    fx_chain = list(_params_dict.keys())
    fx_channel = tc.create_channel(fx_chain)

    params_dict = normalize_param_dict(_params_dict, fx_channel) #normalizing 

    # depending on exact json dict output, this will change
    params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
    if in_sig.batch_size != 1:
        params_list = params_list.transpose(0, 1)
    params = params_list.expand(in_sig.batch_size, -1).to(DEVICE) #shape = (n_batch, n_params)

    out_sig = fx_channel(in_sig.clone().to(DEVICE), params).normalize(-24)
    tc.export_sig(out_sig.clone().detach().cpu(), export_path)
    return out_sig

# def apply_effects_to_sig(audio_dir_or_file: Union[str, Path, List[str], List[Path]], params_dict_path: Union[str, Path], export_path: str):

#     if isinstance(audio_dir_or_file, (str, Path)):
#         audio_dir_or_file = Path(audio_dir_or_file)
#         if audio_dir_or_file.is_dir():
#             print(f'{audio_dir_or_file} is directory')
#             in_sig = tc.wavs_to_batch(audio_dir_or_file)
#         else:
#             print(f'{audio_dir_or_file} is a file')
#             in_sig = tc.preprocess_audio(audio_dir_or_file).to(DEVICE)
#     else:
#         print(f'{audio_dir_or_file} is a list of files')
#         in_sig = tc.wavs_to_batch(audio_dir_or_file)

#     with open(params_dict_path, 'r') as f:
#         params_dict = json.load(f)

#     fx_chain = list(params_dict.keys())
#     fx_channel = tc.create_channel(fx_chain)

#     params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
#     if in_sig.batch_size != 1:
#         params_list = params_list.transpose(0, 1)
#     breakpoint()
#     params = params_list.expand(in_sig.batch_size, -1).to(DEVICE)

#     out_sig = fx_channel(in_sig.clone().to(DEVICE), torch.sigmoid(params))

#     tc.export_sig(out_sig.clone().detach().cpu(), export_path)
    
#     return out_sig


# # testing single file
# apply_effects_to_sig(
#     audio_dir_or_file='assets/multistem_examples/10s/drums.wav',
#     params_dict_path='experiments/2024-07-10/optimize_labeling/process_files_batch/2_tinny_vocals.json',
#     export_path='experiments/2024-07-10/applyFXfromparams/single_2.wav'
# )

# testing multiple files
# apply_effects_to_sig(
#     audio_dir_or_file=['assets/multistem_examples/10s/drums.wav']*3,
#     params_dict_path='experiments/2024-07-10/optimize_labeling/process_files_batch/output_all.json',
#     export_path='experiments/2024-07-10/applyFXfromparams/wav_dir_multi_json2'
# )
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

