from pathlib import Path
from typing import Union, List, Optional

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.__main__ import text2fx
from text2fx.constants import SAMPLE_RATE, DEVICE
"""
Script to process a single audio file with a given FX chain to match a description.
Optional arguments include learning rate, number of steps, loss type, parameter initialization, and augmentation params.
The script saves:
- A dictionary of optimized effect controls as JSON (specified by export_param_dict_path).
- Optionally, an optimized audio file (saved to export_audio_path).

Example Call:
python process_file.py assets/multistem_examples/10s/piano.wav reverb eq compressor smooth \
    --export_param_dict_path experiments/2024-07-10/flatten_list/test1/output.json \
    --export_audio_path experiments/2024-07-10/flatten_list/test1/final_audio.wav \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 600 \
    --criterion cosine-sim \
    --model ms_clap
"""


def main(audio_path: Union[str, Path, AudioSignal], 
         fx_chain: List[str], 
         text_target: str, 
         export_param_dict_path: Optional[str] = None, 
         export_audio_path: Optional[str] = None,
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap') -> dict:
    
    # Preprocess audio, return AudioSignal
    in_sig = tc.preprocess_audio(audio_path).to(DEVICE)
    print(f'1. processed ... {audio_path}')

    # Create FX channel
    fx_channel = tc.create_channel(fx_chain)
    print(f'2. created channel from {fx_chain} ... {fx_channel.modules}')

    # Apply text-to-FX processing
    print(f'3. applying text2fx ...')
    signal_effected, out_params_dict = text2fx(
        model_name=model, 
        sig=in_sig, 
        text=text_target, 
        channel=fx_channel,
        criterion=criterion, 
        params_init_type=params_init_type,
        lr=learning_rate,
        n_iters=n_iters,
        roll_amt=roll_amt,
    )
    # print(f'4. output params {sig_effected_params} ...')
    # Extracting params to dictionary
    # out_params_dict = fx_channel.save_params_to_dict(sig_effected_params)
    # print(f'5. back to dict {out_params_dict} ...')

    # Optionally export optimized parameters as JSON
    if export_param_dict_path:
        print(f'saving final param json to {export_param_dict_path}')
        tc.save_dict_to_json(out_params_dict, export_param_dict_path)

    # Optionally export optimized audio
    if export_audio_path:
        print(f'saving final audio .wav to {export_audio_path}')
        tc.export_sig(signal_effected, export_audio_path)
 
    return out_params_dict

# if __name__ == "__main__":
#     main(
#         audio_path='assets/multistem_examples/10s/guitar.wav',
#         fx_chain=['EQ', 'reverb'],
#         text_target='warm',
#         export_param_dict_path='/home/annie/research/text2fx/experiments/2024-07-08/process_file_test/output.json',
#         export_audio_path='/home/annie/research/text2fx/experiments/2024-07-08/process_file_test/final_audio.wav',
#         learning_rate=1e-2,
#         n_iters=600,
#         params_init_type='random',
#         roll_amt=10000,
#         criterion='cosine-sim',
#         model='ms_clap'
#     )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process an audio file with a given FX chain to match a description.")
    
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    parser.add_argument("fx_chain", nargs="+", help="List of FX to apply.")
    parser.add_argument("text_target", type=str, default='warm', help="Text description to match.")
    parser.add_argument("--export_param_dict_path", type=str, default=None, help="Path to save optimized effect controls as JSON.")
    parser.add_argument("--export_audio_path", type=str, default=None, help="Path to save optimized audio file.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization.")
    parser.add_argument("--params_init_type", type=str, default='random', choices=['random', 'default'], help="Parameter initialization type.")
    parser.add_argument("--roll_amt", type=int, default=None, help="Amount to roll.")
    parser.add_argument("--n_iters", type=int, default=20, help="Number of optimization iterations.")
    parser.add_argument("--criterion", type=str, default='cosine-sim', help="Optimization criterion.")
    parser.add_argument("--model", type=str, default='ms_clap', help="Model name.")

    args = parser.parse_args()

    main(args.audio_path, args.fx_chain, args.text_target,
         args.export_param_dict_path, args.export_audio_path,
         args.learning_rate, args.params_init_type, args.roll_amt,
         args.n_iters, args.criterion, args.model)


 # python process_file.py <audio_path> <fx_chain> <text_target> [--export_param_dict_path EXPORT_PARAM_DICT_PATH]
#                      [--export_audio_path EXPORT_AUDIO_PATH]
#                      [--learning_rate LEARNING_RATE]
#                      [--params_init_type {random,default}]
#                      [--roll_amt ROLL_AMT] [--n_iters N_ITERS]
#                      [--criterion CRITERION] [--model MODEL]


# python process_file.py assets/audio.wav EQ reverb --export_param_dict_path /path/to/save/params.json --export_audio_path /path/to/save/audio.wav --learning_rate 0.001 --params_init_type random --roll_amt 10000 --n_iters 50 --criterion cosine-sim --model ms_clap
