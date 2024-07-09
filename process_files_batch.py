import os
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

from audiotools import AudioSignal
import text2fx.core as tc
from text2fx.__main__ import text2fx
from process_file import main as process_file_main
from itertools import product
from text2fx.constants import SAMPLE_RATE, DEVICE
import json
import torch
AUDIO_SAMPLES_DIR = Path('assets/multistem_examples/10s')
SAMPLE_WORD_LIST = ['happy', 'sad', 'cold']

"""
python process_files_batch.py \
    assets/multistem_examples/10s \
    word_descriptors.txt \
    3 \
    2 \
    EQ reverb \
    --export_dir experiments/2024-07-09/checking_process_files_batch \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 600 \
    --criterion cosine-sim \
    --model ms_clap
    """

def main(audio_dir: Union[str, Path],
         descriptions_source: Union[str, List[str]],
         n_samples: int,
         n_words: int,
         fx_chain: List[str],
         export_dir: Union[str, Path],
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap'):
    
    sampled_audio_files = tc.sample_audio_files(audio_dir, n_samples)
    in_sig_batch = tc.wavs_to_batch(sampled_audio_files)
    sampled_descriptions = tc.sample_words(descriptions_source, n_words)

    fx_channel = tc.create_channel(fx_chain)

    ALL_out_sigs = {}
    ALL_out_params = {}
    check = {}

    for i, description in enumerate(sampled_descriptions):
        out_sig, out_params = text2fx(
            model_name=model,
            sig=in_sig_batch.to(DEVICE),
            channel=fx_channel,
            text=description,
            lr=learning_rate,
            params_init_type=params_init_type,
            roll_amt=roll_amt,
            n_iters=n_iters,
            criterion=criterion,
        )
        out_params_dict = fx_channel.save_params_to_dict(out_params)


        ALL_out_params[description]  = out_params_dict
        ALL_out_sigs[description]  = out_sig
        check[description] = (list(map(str, out_sig.path_to_input_file)), tc.detensor_dict(out_params_dict))
        
        if export_dir is not None:
            # splitting by word
            save_dir = Path(export_dir) / f'{description}'
            tc.save_sig_batch(out_sig, save_dir)
            tc.save_params_batch_to_jsons(out_params_dict, save_dir, out_sig_to_match=out_sig)

            export_param_dict_path = save_dir / f'output_{description}.json'
            tc.save_dict_to_json(out_params_dict, export_param_dict_path)

            export_check_txt_path = save_dir / f'check_{description}.json'
            with open(export_check_txt_path, 'w') as f:
                json.dump(check[description], f, indent=4)  # Serialize to JSON format

    return ALL_out_sigs, ALL_out_params, check

# if __name__ == "__main__":
#     main(
#         audio_dir ='assets/multistem_examples/10s', 
#         descriptions_source = ['happy', 'sad', 'warm', 'cold'],
#         n_samples = 3,
#         n_words = 2,
#         fx_chain = ['compressor', 'reverb'],
#         export_dir = 'experiments/7-09-2024/multipath3',
#         n_iters= 50,
#         criterion = 'cosine-sim',
#         model= 'ms_clap')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple audio files based on sampled descriptions.')
    parser.add_argument('audio_dir', type=str, help='Directory containing audio files.')
    parser.add_argument('descriptions_source', type=str, help='Path to file containing descriptions or a list of descriptions.')
    parser.add_argument('n_samples', type=int, help='Number of audio files to sample.')
    parser.add_argument('n_words', type=int, help='Number of descriptions to sample.')
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
        n_samples=args.n_samples,
        n_words=args.n_words,
        fx_chain=args.fx_chain,
        export_dir=args.export_dir,
        learning_rate=args.learning_rate,
        params_init_type=args.params_init_type,
        roll_amt=args.roll_amt,
        n_iters=args.n_iters,
        criterion=args.criterion,
        model=args.model
    )