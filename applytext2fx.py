from pathlib import Path
from typing import Union, List, Optional

from audiotools import AudioSignal

import text2fx.core as tc
from text2fx.__main__ import text2fx
# from text2fx.production import text2fx_real as text2fx

from text2fx.constants import SAMPLE_RATE, DEVICE

"""
Script to process a single audio file with a given FX chain to match a description.
Optional arguments include learning rate, number of steps, loss type, parameter initialization, and augmentation params.
The script saves:
- A dictionary of optimized effect controls as JSON (specified by export_dir).
- Exported optimized audio file (saved to export_dir).

Example Call:
python applytext2fx.py assets/multistem_examples/10s/bass.wav eq tinny \
    --export_dir experiments/2025-01-28/bass \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 100 \
    --criterion cosine-sim \
    --model ms_clap


case 1 (sparse): single audio file, single text_target
python applytext2fx.py assets/multistem_examples/10s/guitar.wav eq reverb compression 'cold and dark' \
    --export_dir experiments/2025-01-28/guitar_multifx_2 \
    --params_init_type random \
    --n_iters 200 

case 2: multiple audio files, single text_target


case 3: multiple audio file, multiple text_targets (must have same # of files to targets)
 
"""


def main(audio_path: Union[str, Path, AudioSignal], 
         fx_chain: List[str], 
         text_target: str, 
         export_dir: str,
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap') -> dict:

    # Preprocess full audio from path, return AudioSignal
    # in_sig = tc.preprocess_audio(audio_path).to(DEVICE)
    # print(f'1. processing input ... {audio_path}')

        # OPTIONAL: # Preprocess full audio from path, return AudioSignal
    sig_short = AudioSignal.salient_excerpt(audio_path, duration=3).to(DEVICE)
    in_sig = tc.preprocess_audio(sig_short).to(DEVICE)
    print(f'1. processing SHORT (3s) input ... {audio_path}')

    # Create FX channel
    fx_channel = tc.create_channel(fx_chain)
    print(f'2. created channel from {fx_chain} ... {fx_channel.modules}')

    # Apply text-to-FX processing
    print(f'3. applying text2fx ..., target {text_target}')
    signal_effected, out_params, out_params_dict = text2fx(
        model_name=model, 
        sig=in_sig, 
        text=text_target, 
        channel=fx_channel,
        criterion=criterion, 
        save_dir=export_dir,
        params_init_type=params_init_type,
        lr=learning_rate,
        n_iters=n_iters,
        roll_amt=roll_amt,
    )

    # Export JSON parameters & output audio
    if export_dir:
        export_dir = Path(export_dir)
        print(f'saving to {export_dir}')
        json_path = export_dir / f'{text_target}.json'
        print(f'saving final param json to {json_path}')
        tc.save_dict_to_json(out_params_dict, json_path)

        audio_path = export_dir / f'{text_target}.wav'
        print(f'saving final audio .wav to {audio_path}')
        tc.export_sig(signal_effected, audio_path)

        audio_path_in = export_dir / 'input.wav'
        print(f'saving initial audio .wav to {audio_path_in}')
        tc.export_sig(in_sig, audio_path_in)
 
    return out_params_dict


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
         args.model)
