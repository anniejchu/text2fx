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
"""

"""
python applytext2fx_batch.py \
    assets/multistem_examples/10s \
    word_descriptors.txt \
    3 \
    EQ reverb \
    --export_dir experiments/2025-01-28/batch_test \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 50 \
    --criterion cosine-sim \
    --model ms_clap
    """

def main(audio_dir: Union[str, Path],
         descriptions_source: Union[str, List[str]],
         n: int,
         fx_chain: List[str],
         export_dir: Union[str, Path],
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap'):
    
    sampled_audio_files = tc.sample_audio_files(audio_dir, n)
    in_sig_batch = tc.wavs_to_batch(sampled_audio_files)
    sampled_descriptions = tc.sample_words(descriptions_source, n)

    print(sampled_audio_files, sampled_descriptions)
    fx_channel = tc.create_channel(fx_chain)

    signal_effected, out_params, out_params_dict = text2fx(
        model_name=model, 
        sig_in=in_sig_batch, 
        text=sampled_descriptions, 
        channel=fx_channel,
        criterion=criterion, 
        params_init_type=params_init_type,
        lr=learning_rate,
        n_iters=n_iters,
        roll_amt=roll_amt,
    )
    # out_params_dict = fx_channel.save_params_to_dict(sig_effected_params)

    data_labels = list(zip(sampled_audio_files, sampled_descriptions))
    print(data_labels)
    if export_dir is not None:
        print(f'saving final audio .wav to {export_dir}')
        tc.export_sig(signal_effected, export_dir, text=sampled_descriptions)

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
    parser.add_argument('audio_dir', type=str, help='Directory containing audio files.')
    parser.add_argument('descriptions_source', type=str, help='Path to file containing descriptions or a list of descriptions.')
    parser.add_argument('n', type=int, help='Number of audio files to sample.')
    parser.add_argument('fx_chain', type=str, nargs='+', help='List of FX chain elements.')
    parser.add_argument('--export_dir', type=str, default = None, help='Directory to save processed outputs.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimization.')
    parser.add_argument('--params_init_type', type=str, default='random', help='Parameter initialization type.')
    parser.add_argument('--roll_amt', type=int, default=None, help='Roll amount for augmentation.')
    parser.add_argument('--n_iters', type=int, default=50, help='Number of iterations for optimization.')
    parser.add_argument('--criterion', type=str, default='cosine-sim', help='Loss criterion for optimization.')
    parser.add_argument('--model', type=str, default='ms_clap', help='Model name for text-to-FX processing.')

    args = parser.parse_args()
    main(
        audio_dir=args.audio_dir,
        descriptions_source=args.descriptions_source,
        n=args.n,
        fx_chain=args.fx_chain,
        export_dir=args.export_dir,
        learning_rate=args.learning_rate,
        params_init_type=args.params_init_type,
        roll_amt=args.roll_amt,
        n_iters=args.n_iters,
        criterion=args.criterion,
        model=args.model
    )