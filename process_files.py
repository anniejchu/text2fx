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

AUDIO_SAMPLES_DIR = Path('assets/multistem_examples/10s')
SAMPLE_WORD_LIST = ['happy', 'sad', 'cold']

"""
python process_files.py \
    assets/multistem_examples/10s \
    word_descriptors.txt \
    3 \
    2 \
    EQ reverb \
    experiments/2024-07-08/process_FILES_test \
    --learning_rate 0.01 \
    --params_init_type random \
    --roll_amt 10000 \
    --n_iters 600 \
    --criterion cosine-sim \
    --model ms_clap
    """
def sample_audio_files(audio_dir: Union[str, Path], n: int) -> List[Path]:
    """
    Samples n audio paths and n descriptions from the given directories.

    :param audio_dir: Directory containing audio files.
    :param descriptions_file: File containing descriptions.
    :param n: Number of samples to process.
    :return: List of tuples containing (audio_path, description).
    """
    if isinstance(audio_dir, str):
        audio_dir=Path(audio_dir)
    audio_files = tc.load_examples(audio_dir)

    if len(audio_files) < n:# or len(descriptions) < n:
        raise ValueError("Not enough audio files to sample from")

    sampled_audio_files = random.sample(audio_files, n)
    return sampled_audio_files 

def sample_words(words_source: Union[str, Path, List[str]], n: int) -> List[str]:
    """
    Samples n words from the given word descriptor source.

    :param words_source: File containing word descriptors (one per line) or a list of descriptors.
    :param n: Number of descriptor words to sample.
    :return: List of sampled descriptor words.
    """
    if isinstance(words_source, (str, Path)):
        with open(words_source, 'r') as f:
            word_list = [line.strip() for line in f if line.strip()]
    else:
        word_list = words_source

    if len(word_list) < n:
        raise ValueError("Not enough descriptions to sample from")

    sampled_words = random.sample(word_list, n)
    return sampled_words


def main(audio_dir: Union[str, Path],
         descriptions_source: Union[str, List[str]],
         n_samples: int,
         n_words: int,
         fx_chain: List[str],
         export_dir: Union[str, Path],
         learning_rate: float = 0.001,
         params_init_type: str = 'random',
         roll_amt: Optional[int] = None,
         n_iters: int = 50,
         criterion: str = 'cosine-sim',
         model: str = 'ms_clap'):
    
    sampled_audio_files = sample_audio_files(audio_dir, n_samples)
    sampled_descriptions = sample_words(descriptions_source, n_words)

    for i, (audio_path, description) in enumerate(product(sampled_audio_files, sampled_descriptions)):
        export_param_dict_path = Path(export_dir) / f'output_{i}_{audio_path.stem}_{description}.json'
        export_audio_path = Path(export_dir) / f'final_audio_{i}_{audio_path.stem}_{description}.wav'

        print(f' ------ text2fx-ing {audio_path.stem}, {description}')
        process_file_main(
            audio_path=audio_path,
            fx_chain=fx_chain,
            text_target=description,
            export_param_dict_path=str(export_param_dict_path),
            export_audio_path=str(export_audio_path),
            learning_rate=learning_rate,
            params_init_type=params_init_type,
            roll_amt=roll_amt,
            n_iters=n_iters,
            criterion=criterion,
            model=model
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple audio files based on sampled descriptions.')
    parser.add_argument('audio_dir', type=str, help='Directory containing audio files.')
    parser.add_argument('descriptions_source', type=str, help='Path to file containing descriptions or a list of descriptions.')
    parser.add_argument('n_samples', type=int, help='Number of audio files to sample.')
    parser.add_argument('n_words', type=int, help='Number of descriptions to sample.')
    parser.add_argument('fx_chain', type=str, nargs='+', help='List of FX chain elements.')
    parser.add_argument('export_dir', type=str, help='Directory to save processed outputs.')
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