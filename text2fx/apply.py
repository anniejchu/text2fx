from pathlib import Path
from typing import Union, List, Optional, Tuple
import torch

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.__main__ import text2fx
from text2fx.constants import SAMPLE_RATE, DEVICE

"""
Script to process a single audio file with a given FX chain to match a description.
Optional arguments include learning rate, number of steps, loss type, parameter initialization, and augmentation params.
The script saves:
- A dictionary of optimized effect controls as JSON (specified by export_dir).
- Exported optimized audio file (saved to export_dir).

Example Call:
from text2fx not text2fx/text2fx

python -m text2fx.apply assets/multistem_examples/10s/bass.wav eq tinny \
    --export_dir experiments/2025-02-18/prod_detailedlog2 \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 400 \
    --criterion cosine-sim \
    --model ms_clap \
    --detailed_log

python -m text2fx.apply assets/multistem_examples/10s/bass.wav eq tinny \
    --export_dir experiments/2025-01-28/bass \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 100 \
    --criterion cosine-sim \
    --model ms_clap


    
case 1 (sparse): single audio file, single text_target
python -m text2fx.apply assets/multistem_examples/10s/guitar.wav eq reverb compression 'cold and dark' \
    --export_dir experiments/2025-01-28/guitar_multifx_2 \
    --params_init_type random \
    --n_iters 200 

python -m text2fx.apply /home/annie/research/text2fx/assets/multistem_examples/10s/guitar.wav eq 'cold and dark' \
--export_dir experiments/2025-02-17/prod/single \
--params_init_type random \
--n_iters 200 
"""


def main(audio_path: Union[str, Path, AudioSignal], 
         fx_chain: List[str], 
         text_target: str, 
         export_dir: str = None,
         learning_rate: float = 0.01,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap',
         detailed_log:bool = False) -> Tuple[AudioSignal, torch.Tensor, dict]:

    # Preprocess full audio from path, return AudioSignal
    print('text2fx on full sig')
    in_sig = tc.preprocess_audio(audio_path).to(DEVICE)

    # print('text2fx on 3s salient_excerpt')
    # in_sig = tc.preprocess_audio(audio_path, salient_excerpt_duration=3).to(DEVICE)

    print(f'1. processing input ... {audio_path}')

    # Create FX channel
    fx_channel = tc.create_channel(fx_chain)
    print(f'2. created channel from {fx_chain} ... {fx_channel.modules}')

    # Apply text-to-FX processing
    print(f'3. applying text2fx ..., target {text_target}')
    if detailed_log:
        print('with detailed logging every 100 iters')
    
        # Export JSON parameters & output audio
    if export_dir:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        save_dir = tc.create_save_dir(f'{text_target}', Path(export_dir))
    else:
        save_dir = None
        
    signal_effected, out_params, out_params_dict = text2fx(
        model_name=model, 
        sig_in=audio_path, 
        text=text_target, 
        channel=fx_channel,
        criterion=criterion, 
        save_dir=save_dir,
        params_init_type=params_init_type,
        lr=learning_rate,
        n_iters=n_iters,
        roll_amt=roll_amt,
        detailed_log=detailed_log,
    )

    if export_dir:
        json_path = save_dir / 'FXparams.json'
        print(f'saving final param json to {json_path}')
        tc.save_dict_to_json(out_params_dict, json_path)

        audio_path = save_dir / 'output.wav'
        print(f'saving final audio .wav to {audio_path}')
        tc.export_sig(signal_effected, audio_path)

        audio_path_in = save_dir / 'input_to_text2fx.wav'
        print(f'saving initial audio .wav to {audio_path_in}')
        tc.export_sig(tc.preprocess_audio(in_sig), audio_path_in)

    return signal_effected, out_params, out_params_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process an audio file with a given FX chain to match a description.")
    
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    parser.add_argument("fx_chain", nargs="+", help="List of FX to apply.")
    parser.add_argument("text_target", type=str, default='warm', help="Text description to match.")
    parser.add_argument("--export_dir", type=str, default=None, help="Dir Path to save optimized audio file.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimization.")
    parser.add_argument("--params_init_type", type=str, default='random', choices=['random', 'default'], help="Parameter initialization type.")
    parser.add_argument("--roll_amt", type=int, default=None, help="Amount to roll.")
    parser.add_argument("--n_iters", type=int, default=600, help="Number of optimization iterations.")
    parser.add_argument("--criterion", type=str, default='cosine-sim', help="Optimization criterion.")
    parser.add_argument("--model", type=str, default='ms_clap', help="Model name.")
    parser.add_argument("--detailed_log", action="store_true", help="Enable detailed logging every 100 iterations.")


    args = parser.parse_args()

    main(args.audio_path, 
         args.fx_chain, 
         args.text_target,
         args.export_dir,
         args.learning_rate, 
         args.params_init_type, 
         args.roll_amt,
         args.n_iters, 
         args.criterion, 
         args.model,
         args.detailed_log)
