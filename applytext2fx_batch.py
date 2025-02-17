import os
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

from audiotools import AudioSignal
import text2fx.core as tc
from text2fx.__main__ import text2fx
from applytext2fx import main as process_file_main
from itertools import product
from text2fx.constants import SAMPLE_RATE, DEVICE
import json
import torch

# AUDIO_SAMPLES_DIR = Path('assets/multistem_examples/10s')
# SAMPLE_WORD_LIST = ['happy', 'sad', 'cold']

"""
TODOS
1. simplify & remove sampling by n, do #assert len(batch_audio) == len(batch_texts)
2. what cases

case 1: multiple audio files, single text_target

case 2: single audio file, list of words
"""

"""
case 2: single audio file, list of words


case 1: multiple audio files, single text_target
python applytext2fx_batch.py \
    --audio_dir assets/multistem_examples/10s \
    --descriptions_source "cold" \
    --fx_chain eq \
    --export_dir experiments/2025-02-17/batch_test_batchaudio_singledescriptor \
    --learning_rate 0.01 \
    --params_init_type random \
    --n_iters 50 \
    --criterion cosine-sim \
    --model ms_clap

case 3: multiple audio file, multiple text_targets (must have same # of files to targets)
python applytext2fx_batch.py \
    --audio_dir assets/multistem_examples/10s \
    --descriptions_source "cold, warm, like a trumpet, muffled, lonely like a ghost" \
    --fx_chain eq reverb \
    --export_dir experiments/2025-02-17/batch_test \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 50 \
    --criterion cosine-sim \
    --model ms_clap
    """

# # case 2: single audio file, list of words
# def clone_single_sig(audio_path: Union[str, Path],
#               descriptions_source: Union[str, List[str]]):
#     sig = AudioSignal(audio_path)
    
#     signal_list = [sig for _ in range(len(descriptions_source))]
#     sig_batch = AudioSignal.batch(signal_list)
#     print(sig_batch)
#     return sig_batch


# # case 3: multiple audio files, single text_target
# def multicopy_descriptors():
#     pass


def main(audio_dir: Union[str, Path],
         descriptions_source: Union[str, List[str]],
         fx_chain: List[str],
         export_dir: Union[str, Path],
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap'):
    
    audio_file_paths = tc.load_examples(audio_dir)
    in_sig_batch = tc.wavs_to_batch(audio_file_paths)
    descriptor_list = tc.load_words(descriptions_source)

    print(in_sig_batch)
    print(descriptor_list)
    assert in_sig_batch.batch_size == len(descriptor_list) or len(descriptor_list) == 1

    print(audio_file_paths, descriptor_list)
    breakpoint()
    
    fx_channel = tc.create_channel(fx_chain)
    print(fx_channel)

    breakpoint()
    signal_effected, out_params, out_params_dict = text2fx(
        model_name=model, 
        sig_in=in_sig_batch, 
        text=descriptor_list, 
        channel=fx_channel,
        criterion=criterion, 
        params_init_type=params_init_type,
        lr=learning_rate,
        n_iters=n_iters,
        roll_amt=roll_amt,
    )
    # out_params_dict = fx_channel.save_params_to_dict(sig_effected_params)
    breakpoint()
    if len(descriptor_list) == 1:
    # Repeat the single descriptor to match the length of audio_file_paths
        descriptor_list = [descriptor_list[0]] * len(audio_file_paths)

    data_labels = list(zip(audio_file_paths, descriptor_list))
    print(data_labels)
    breakpoint()

    if export_dir is not None:
        print(f'saving final audio .wav to {export_dir}')
        tc.export_sig(signal_effected, export_dir, text=descriptor_list)

        export_param_dict_path = Path(export_dir) / f'output_all.json'
        print(f'saving final param json to {export_param_dict_path}')
        tc.save_dict_to_json(out_params_dict, export_param_dict_path)
        tc.save_params_batch_to_jsons(out_params_dict, export_dir, data_labels=data_labels)

    return out_params_dict


# if __name__ == "__main__":
#     main(
#         audio_dir ='assets/multistem_examples/10s', 
#         descriptions_source = ['happy', 'sad', 'warm', 'cold'],
#         n = 3,
#         fx_chain = ['compressor', 'reverb'],
#         export_dir = 'experiments/2024-07-09/batched_text',
#         params_init_type='zeros',
#         n_iters= 50,
#         criterion = 'cosine-sim',
#         model= 'ms_clap')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process multiple audio files based on sampled descriptions.')
    parser.add_argument('--audio_dir', type=str, default=None, help='Directory containing audio files.')
    parser.add_argument('--descriptions_source', type=str, default=None, help='Comma-separated list of descriptions.')
    parser.add_argument('--fx_chain', type=str, nargs='+', default='eq', help='List of FX chain elements.')
    parser.add_argument('--export_dir', type=str, default = None, help='Directory to save processed outputs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization.')
    parser.add_argument('--params_init_type', type=str, default='random', help='Parameter initialization type.')
    parser.add_argument('--roll_amt', type=int, default=None, help='Roll amount for augmentation.')
    parser.add_argument('--n_iters', type=int, default=50, help='Number of iterations for optimization.')
    parser.add_argument('--criterion', type=str, default='cosine-sim', help='Loss criterion for optimization.')
    parser.add_argument('--model', type=str, default='ms_clap', help='Model name for text-to-FX processing.')

    args = parser.parse_args()
    descriptions = [desc.strip() for desc in args.descriptions_source.split(',')] if args.descriptions_source else []
    print(descriptions)

    main(
        audio_dir=args.audio_dir,
        descriptions_source=descriptions,#args.descriptions_source,
        fx_chain=args.fx_chain,
        export_dir=args.export_dir,
        learning_rate=args.learning_rate,
        params_init_type=args.params_init_type,
        roll_amt=args.roll_amt,
        n_iters=args.n_iters,
        criterion=args.criterion,
        model=args.model
    )