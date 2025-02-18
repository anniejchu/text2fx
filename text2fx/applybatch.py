import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
from audiotools import AudioSignal
import text2fx.core as tc
from text2fx.__main__ import text2fx
from text2fx.constants import SAMPLE_RATE, DEVICE


"""
case 1: multiple audio files, single text_target
python -m text2fx.applybatch \
    --audio_source assets/multistem_examples/10s \
    --descriptions_source "cold" \
    --fx_chain eq \
    --export_dir experiments/2025-02-17/prod/batch_test_batchaudio_singledescriptor2 \
    --learning_rate 0.01 \
    --params_init_type random \
    --n_iters 50 \
    --criterion cosine-sim \
    --model ms_clap

case 2: single audio file, list of words
python -m text2fx.applybatch \
    --audio_source assets/multistem_examples/10s/drums.wav \
    --descriptions_source "cold, warm, like a trumpet, muffled, lonely like a ghost" \
    --fx_chain eq \
    --export_dir experiments/2025-02-17/prod/batch_test_singlesig_multidescriptor \
    --learning_rate 0.01 \
    --params_init_type random \
    --n_iters 50 \
    --criterion cosine-sim \
    --model ms_clap


case 3: multiple audio file, multiple text_targets (must have same # of files to targets)
python -m text2fx.applybatch \
    --audio_source assets/multistem_examples/10s \
    --descriptions_source "cold, warm, like a trumpet, muffled, lonely like a ghost" \
    --fx_chain eq reverb \
    --export_dir experiments/2025-02-17/prod/batch_test2 \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 50 \
    --criterion cosine-sim \
    --model ms_clap
    """


def main(audio_source: Union[str, Path], #can be path to single file or dir of files
         descriptions_source: Union[str, List[str]],
         fx_chain: List[str],
         export_dir: Union[str, Path],
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 600,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap') -> Tuple[AudioSignal, torch.Tensor, dict]:
    
    audio_file_paths = tc.load_examples(audio_source)
    descriptor_list = tc.load_words(descriptions_source)

    if len(audio_file_paths) > 1:
        in_sig_batch = tc.wavs_to_batch(audio_file_paths)
    else:
        sig = AudioSignal(Path(audio_source))
        signal_list = [sig for _ in range(len(descriptor_list))]
        in_sig_batch = AudioSignal.batch(signal_list)

    print(in_sig_batch)
    print(descriptor_list)

    print(in_sig_batch.batch_size, len(descriptor_list))

    #at this point, in_sig_batch must have same length or descriptor list or only one word
    assert in_sig_batch.batch_size == len(descriptor_list) or len(descriptor_list) == 1

    print(f'audio source paths: {audio_file_paths}, descriptor: {descriptor_list}')
    
    fx_channel = tc.create_channel(fx_chain)

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
    if len(descriptor_list) == 1 and len(audio_file_paths) != 1:
    # Repeat the single descriptor to match the length of audio_file_paths
        descriptor_list = [descriptor_list[0]] * len(audio_file_paths)

    elif len(audio_file_paths) == 1 and len(descriptor_list) != 1:
        audio_file_paths = [audio_file_paths[0]] * len(descriptor_list)


    assert len(audio_file_paths) == len(descriptor_list)
    data_labels = list(zip(audio_file_paths, descriptor_list))

    if export_dir is not None:
        print(f'saving final audio .wav to {export_dir}')
        tc.export_sig(signal_effected, export_dir, text=descriptor_list)

        export_param_dict_path = Path(export_dir) / f'output_all.json'
        print(f'saving final param json to {export_param_dict_path}')
        tc.save_dict_to_json(out_params_dict, export_param_dict_path)
        tc.save_params_batch_to_jsons(out_params_dict, export_dir, data_labels=data_labels)

    return signal_effected, out_params, out_params_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process multiple audio files based on sampled descriptions.')
    parser.add_argument('--audio_source', type=str, default=None, help='Path to file or Directory containing audio files.')
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
        audio_source=args.audio_source,
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