from pathlib import Path
import datetime
import unicodedata
import re
import random
import json
import os
from pathlib import Path
from typing import Union, List, Optional, Tuple, Iterable, Dict

import torch
import torchaudio.transforms as T
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

import audiotools as at
from audiotools import AudioSignal
import dasp_pytorch
import auraloss
from functools import partial

from text2fx.constants import PROJECT_DIR, ASSETS_DIR, PRETRAINED_DIR, DATA_DIR, RUNS_DIR, EQ_freq_bands, SAMPLE_RATE, EQ_GAINS_PATH
from dasp_pytorch.modules import normalize
"""
EX CLI USAGE
python text2fx.py --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
                 --text "this sound is happy" \
                 --criterion "cosine-sim" \
                 --n_iters 600 \
                 --lr 0.01 
"""

class AbstractCLAPWrapper:
    def preprocess_audio(self, signal: AudioSignal) -> AudioSignal:
        raise NotImplementedError("implement me :)")
    
    def get_audio_embeddings(self, signal: AudioSignal) -> torch.Tensor:
        raise NotImplementedError()
    
    def get_text_embeddings(self, text: Union[str, List[str]]) -> torch.Tensor:
        raise NotImplementedError()
    
    def compute_similarities(self, audio_emb, text_emb) -> torch.Tensor:
        raise NotImplementedError
    
    @property
    def sample_rate(self):
        raise NotImplementedError()


class Distortion(dasp_pytorch.modules.Processor):
    def __init__(
        self,
        sample_rate: int = None,
        min_drive_db: float = 0.0,
        max_drive_db: float = 24.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.process_fn = dasp_pytorch.functional.distortion
        self.param_ranges = {
            "drive_db": (min_drive_db, max_drive_db),
        }
        self.num_params = len(self.param_ranges)


class Channel(torch.nn.Module):
    def __init__(self, *args):
    
        super().__init__()
    
        modules = []
        if isinstance(args[0], Iterable) and len(args) == 1:
            for m in args[0]:
                assert isinstance(m, dasp_pytorch.modules.Processor)
                modules.append(m)
        else:
            for m in args:
                assert isinstance(m, dasp_pytorch.modules.Processor)
                modules.append(m)

        # Ensure consistent sample rate
        sample_rates = [m.sample_rate for m in modules]

        # If not uniform, go with highest sample rate
        self.sample_rate = max(sample_rates)

        for i, m in enumerate(modules):
            modules[i].sample_rate = self.sample_rate
        self.modules = modules

    @property #hacky thing decorator/annotator, concrete attribute, this is a getter
    def num_params(self):
        return sum([m.num_params for m in self.modules])

    #if you call the object, it automatically calls **forward()** (uses __call__)
    def forward(self, signal: AudioSignal, params: torch.Tensor)  -> AudioSignal:

        output = signal.clone().resample(self.sample_rate)
        
        # Check for valid shape
        assert params.ndim == 2  # (n_batch, n_parameters)
        assert params.shape[-1] == self.num_params
        
        params_count = 0
        for m in self.modules:

            # Select parameters corresponding to current effect module
            _params = params[:, params_count: params_count + m.num_params]
            params_count += m.num_params

            # Apply effect
            output.audio_data = m.process_normalized(output.audio_data, _params) #so assumes _params is normalized [0, 1]

            # Avoid clipping
            output.ensure_max_of_audio()
            
        return output.resample(signal.sample_rate)  # Restore original sample rate

    def save_params_to_dict(self, params: torch.Tensor, save_path:str=None) -> dict:
        """Save parameter tensors for each module to structured dictionaries.

        Args:
            params (torch.Tensor): The parameter tensor.

        Returns:
            dict: Structured dictionary of parameter tensors for each module.
        """
        all_params = {}
        params_count = 0
        for m in self.modules:
            # Extracting respective params for each module in Channel
            _params = params[:, params_count: params_count + m.num_params]
            params_count += m.num_params

            # Normalizing them before extracting param_dict
            _params_min_vals = _params.min(dim=0).values
            _params_max_vals = _params.max(dim=0).values
            _params_normalized = normalize(_params, _params_min_vals, _params_max_vals)

            raw_param_dict = m.extract_param_dict(_params_normalized)
            # breakpoint()
            denorm_param_dict = m.denormalize_param_dict(raw_param_dict)

            # denorm_param_dict = {k: v.tolist() for k, v in denorm_param_dict.items()}
            all_params[m.__class__.__name__] = denorm_param_dict

        # with open(save_path, 'w') as f:
        #     json.dump(all_params, f, indent=4)

        return all_params
    
class ParametricEQ_40band(dasp_pytorch.modules.Processor):
    def __init__(
        self,
        sample_rate: int,
        q_factor: float = 4.31,
        band_freqs: list = EQ_freq_bands, #list
        min_gain_db: float = -20.0,
        max_gain_db: float = 20.0,

    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.q_factor = q_factor
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        # setting band frequencies dynamically
        for i in range(len(band_freqs)):
            setattr(self, f"band{i}_freq", band_freqs[i])

        # setting
        self.process_fn = partial(
            functional_parametric_eq_40band,
            q=q_factor
        )

        # Loop to populate param_ranges, used in denormalize
        self.param_ranges = {}
        for i in range(len(band_freqs)):
            self.param_ranges[f"band{i}_gain_db"] = (min_gain_db, max_gain_db)

        self.num_params = len(self.param_ranges)

def functional_parametric_eq_40band(
        x: torch.Tensor, 
        sample_rate: int, 
        band0_gain_db: torch.Tensor,
        band1_gain_db: torch.Tensor,
        band2_gain_db: torch.Tensor,
        band3_gain_db: torch.Tensor,
        band4_gain_db: torch.Tensor,
        band5_gain_db: torch.Tensor,
        band6_gain_db: torch.Tensor,
        band7_gain_db: torch.Tensor,
        band8_gain_db: torch.Tensor,
        band9_gain_db: torch.Tensor,
        band10_gain_db: torch.Tensor,
        band11_gain_db: torch.Tensor,
        band12_gain_db: torch.Tensor,
        band13_gain_db: torch.Tensor,
        band14_gain_db: torch.Tensor,
        band15_gain_db: torch.Tensor,
        band16_gain_db: torch.Tensor,
        band17_gain_db: torch.Tensor,
        band18_gain_db: torch.Tensor,
        band19_gain_db: torch.Tensor,
        band20_gain_db: torch.Tensor,
        band21_gain_db: torch.Tensor,
        band22_gain_db: torch.Tensor,
        band23_gain_db: torch.Tensor,
        band24_gain_db: torch.Tensor,
        band25_gain_db: torch.Tensor,
        band26_gain_db: torch.Tensor,
        band27_gain_db: torch.Tensor,
        band28_gain_db: torch.Tensor,
        band29_gain_db: torch.Tensor,
        band30_gain_db: torch.Tensor,
        band31_gain_db: torch.Tensor,
        band32_gain_db: torch.Tensor,
        band33_gain_db: torch.Tensor,
        band34_gain_db: torch.Tensor,
        band35_gain_db: torch.Tensor,
        band36_gain_db: torch.Tensor,
        band37_gain_db: torch.Tensor,
        band38_gain_db: torch.Tensor,
        band39_gain_db: torch.Tensor,
        q: float, 
    ) -> torch.Tensor:

    x_out = x.clone()
    
    nb,nc,nt = x_out.shape
    # print(nb, nc, nt)
    x_out= x_out.view(nb*nc,1,nt)
    
    band_freqs = torch.tensor(EQ_freq_bands, device=x.device).reshape(1, -1).repeat(nb*nc, 1) # specify frequencies
    # print(band_freqs)
    qs = torch.tensor([q], device=x.device).reshape(1).repeat(nb*nc)
    # print(qs)
    for i in range(len(band_freqs)):
        band_gain_i = locals()[f"band{i}_gain_db"] 

        # Design peak filter
        b, a = dasp_pytorch.signal.biquad(band_gain_i, band_freqs[:, i], qs, sample_rate, 'peaking')
        
        x_out = dasp_pytorch.signal.lfilter_via_fsm(x_out, b, a)

    x_out= x_out.view(nb,nc,nt) #this should be output

    return x_out # Returns: x_out (torch.Tensor): filtered signal


def dasp_apply_EQ_file(file_name, freqs, Q=4.31): #process function
    """
    file(input signal) = file_name or mono or stereo (bs, n_channels, signals)
                        ex torch.Size([1, 1, 451714])
    filters = list of (frequency, gain_db) pairs
    returns = output AudioSignal, filtered signal as (bs, n_channels, signals)
    """
    audio = AudioSignal(file_name)
    x = audio.samples
    fs = audio.sample_rate

    filtered_sig = functional_parametric_eq_40band(x, fs, freqs, Q)
    out_audiosig = AudioSignal(filtered_sig, fs).ensure_max_of_audio()

    return out_audiosig

def load_examples(dir_path: Path) -> List[Path]:
    exts = ["mp3", "wav", "flac"]
    example_files = [list(dir_path.rglob(f"*.{e}")) for e in exts]
    example_files = sum(example_files, [])  # Trick to flatten list of lists
    return example_files

def download_file(url, out_dir, chunk_size: int = 8192):
    
    local_filename = Path(out_dir) / url.split('/')[-1]

    # Determine download size
    response = requests.get(url, stream=True)
    total_bytes = int(response.headers['Content-length'])
    n_iter = total_bytes // chunk_size
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=n_iter): 
                f.write(chunk)
    return local_filename


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def create_save_dir(text, runs_dir):
    """ 
    Create a save folder for our current run.
    """
    # Create a directory for today's date in YYYY-MM-DD format
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    run_name = f"{slugify(text)}"    
    today_dir = Path(runs_dir) / today
    today_dir.mkdir(parents=True, exist_ok=True)

    # Check if there are any directories with the same run name
    existing_runs = [d for d in today_dir.iterdir() if d.is_dir() and d.name.startswith(run_name)]

    if len(existing_runs) == 0:
        save_dir = today_dir / f"{run_name}-001"
    else:
        latest_run = sorted(existing_runs, key=lambda x: int(x.name.split('-')[-1]))[-1]
        run_num = int(latest_run.name.split('-')[-1]) + 1
        save_dir = today_dir / f"{run_name}-{run_num:03d}"

    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir

#from notebook / helper.py 


def find_paths_with_keyword(file_paths, keywords, returnSingle=False):
    """
    Find paths containing all given keywords.
    
    Args:
    - file_paths (list of str or PosixPath): List of file paths to search.
    - keywords (list of str): List of keywords to search for.
    - return_single (bool): If True, return only the first match. If False, return a list of all matches.
    
    Returns:
    - str or list of str: Single path or list of paths matching the keywords.
    """
    if returnSingle:
        return next((file for file in file_paths if all(keyword in str(file) for keyword in keywords)), None)
    else:
        return [file for file in file_paths if all(keyword in str(file) for keyword in keywords)]

def load_and_find_path_with_keyword(dir_path, keywords, returnSingle=False, exactmatch=False):
    """
    Search for files given a folder (can be nested) and multiple keywords.
    
    Args:
    - dir_path (str or PosixPath): Parent directory to pull all audio files from.
    - keywords (list of str): List of keywords to search for.
    - return_single (bool): If True, return only the first match. If False, return a list of all matches.
    
    Returns:
    - str or list of str: Single path or list of paths matching the keywords.
    """
    examples_all = load_examples(dir_path)
    # if exactmatch:
    #     return find_paths_with_EXACTkeyword(examples_all, keywords, returnSingle=returnSingle)
    # else:
    return find_paths_with_keyword(examples_all, keywords, returnSingle=returnSingle)


def plot_eq_curve(freq_bands, gains):
    """
    Plots the frequency response of the equalizer.
    
    Args:
    - freq_bands: List of center frequencies of the filters.
    - gains: List of gain values for each filter.
    """
    freq_bands = np.array(freq_bands)
    gains = np.array(gains)*5
    
    plt.semilogx(freq_bands, gains, label='Equalizer Curve', marker='o')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain (dB)')
    plt.title('Equalizer Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def find_settings_for_word(data, search_word):
    for item in data:
        if item.get("word") == search_word:
            return item.get("settings", [])
    return []

def get_settings_for_words(file_path, words):
    """
    Load JSON data from a file and find settings for each word in a list.
    
    Parameters:
    - file_path: str, path to the JSON file.
    - words: list of str, words to search for in the JSON data.
    
    Returns:
    - dict: dictionary where keys are words and values are their settings.
    """
    # Load JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Create a dictionary to store the results
    settings_dict = {}
    
    # Loop through each word in the list and find its settings
    for word in words:
        settings_dict[word] = find_settings_for_word(data, word)
    
    return settings_dict

def convert_to_freq_gain_tuples(settings_dict, freq_values):
    """
    Convert gain values to a list of tuples of (frequency, gain).

    Parameters:
    - settings_dict: dict, dictionary containing settings for each word.
    - freq_values: list of int, frequency values.

    Returns:
    - dict: dictionary with settings where gain values are replaced with tuples of (frequency, gain).
    """
    # Create a new dictionary to store the converted settings
    converted_settings = {}

    # Iterate over each word and its corresponding settings
    for word, gain_values in settings_dict.items():
        # Convert gain values to a list of tuples of (frequency, gain)
        freq_gain_tuples = list(zip(freq_values, gain_values))
        # Update the settings for the word with the converted tuples
        converted_settings[word] = freq_gain_tuples

    return converted_settings


def convert_to_tensors(converted_settings):
    """
    Convert all values in the dictionary to tensors.

    Parameters:
    - converted_settings: dict, dictionary with settings where gain values are replaced with tuples of (frequency, gain).

    Returns:
    - dict: dictionary with values converted to tensors.
    """
    # Create a new dictionary to store the converted tensors
    tensor_settings = {}

    # Iterate over each word and its corresponding settings
    for word, freq_gain_tuples in converted_settings.items():
        # Convert each tuple to a tensor
        tensor_freq_gain_tuples = [(torch.tensor([freq]), torch.tensor([gain])) for freq, gain in freq_gain_tuples]
        # Update the settings for the word with the converted tensors
        tensor_settings[word] = tensor_freq_gain_tuples

    return tensor_settings

# probably return tensor instead of audiosig for easier batching
def preprocess_audio(audio_path_or_array: Union[torch.Tensor, str, Path, np.ndarray, AudioSignal], sample_rate: Optional[int] = None) -> AudioSignal:
    #audio can be filename or AudioSignal; if tensor, must provide sample_rate
    if isinstance(audio_path_or_array, (str, Path)):
        return AudioSignal(audio_path_or_array).to_mono().resample(SAMPLE_RATE).normalize(-24)
    elif isinstance(audio_path_or_array, AudioSignal):
        return audio_path_or_array.to_mono().resample(SAMPLE_RATE).normalize(-24)
    elif isinstance(audio_path_or_array, (torch.Tensor, np.ndarray)):
        if sample_rate is None:
            raise ValueError("Must provide sample_rate if input is a tensor or ndarray")
        return AudioSignal(audio_path_or_array, sample_rate).to_mono().resample(SAMPLE_RATE).normalize(-24)
    else: 
        raise ValueError("not audiosignal, tensor, str, path or array")
    
def preprocess_export_audiodir(dir_path, out_dir_path):
    for file in load_examples(dir_path):
        x = preprocess_audio(file)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        x.write(out_dir_path / f'{file.stem}_preprocessed.wav')

def compare_loss_files_preprocessed(file_baseline, file_compare, loss_funct=auraloss.freq.MultiResolutionSTFTLoss()):
    baselineSig = preprocess_audio(file_baseline)
    compareSig = preprocess_audio(file_compare)
    loss = loss_funct(baselineSig.samples, compareSig.samples)
    return loss


def calculate_auraloss_sigs(sig1, sig2, loss_funct=auraloss.freq.MultiResolutionSTFTLoss()):
    loss = loss_funct(sig1.samples, sig2.samples) 
    return loss


def apply_export_EQ(tensor_settings, input_file, export_parent_dir):
    """
    Processes tensor settings to filter audio and save the results.

    Parameters:
    tensor_settings (dict): A dictionary where keys are words and values are frequency gains.
    input_file (InputType): The filename input data to be filtered.
    AUDEALIZE_GND_TRUTH_DIR (Path): The directory where the ground truth audio files are to be saved.

    Returns:
    None
    """
    for word, freq_gains in tensor_settings.items():
        filter_out = dasp_apply_EQ_file(input_file, freq_gains)
        print(f'applying EQ for {word} to -> {input_file.stem}')

        EXPORT_EX_DIR = Path(export_parent_dir / f"{input_file.stem}")
        EXPORT_EX_DIR.mkdir(exist_ok=True)

        filter_out.write(Path(EXPORT_EX_DIR, f"EQed_{word}.wav"))


def find_wav_files(directory: Path) -> List[Tuple[Path, Path, Path]]:
    wav_files = []
    for root, _, files in os.walk(directory):
        if all(f in files for f in ['starting.wav', 'input.wav', 'final.wav']):
            starting_wav = Path(root) / 'starting.wav'
            input_wav = Path(root) / 'input.wav'
            final_wav = Path(root) / 'final.wav'
            wav_files.append((starting_wav, input_wav, final_wav))
    return wav_files

def printy(path_dir: Union[Path, List[Path]]):
    if isinstance(path_dir, list):
        for path in path_dir:
            print(path)
    else:
        print(path_dir)

#
# convert a directory of audio examples to a single batched_AudioSignal
def wav_dir_to_batch(samples_dir) -> AudioSignal:
    all_raw_sigs = load_examples(samples_dir)
    signal_list = [preprocess_audio(raw_sig_i) for raw_sig_i in all_raw_sigs]
    sig_batched = AudioSignal.batch(signal_list)
    return sig_batched

# applying single word EQ params (e.g. 'warm') on a batched AudioSignal
def apply_single_word_EQ_to_batch(signal_batch: AudioSignal, word: str = 'none'):
    m40b = ParametricEQ_40band(sample_rate=44100)
    channel = Channel(m40b)
    freq_gains_dict = get_settings_for_words(EQ_GAINS_PATH, [word])
    # Use GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    signal = signal_batch.to(device)

    if word=='none':
        print(f'... setting random params')
        params = torch.randn(signal.batch_size, channel.num_params).to(device)
    else:
        print(f'...getting parameters for {word}')
        param_single = torch.tensor(freq_gains_dict[word])*5 #make dict parameter
        # print(f'got it: {freq_gains_dict[word]}')

        params = param_single.expand(signal.batch_size, -1).to(device)

    signal_effected_batch = channel(signal, torch.sigmoid(params)).clone().detach().cpu()
    return signal_effected_batch

# applying list of words EQ params (e.g. ['warm', 'cool]) on a batched AudioSignal
def apply_multi_word_EQ_to_batch(signal_batch: AudioSignal, word_list: List[str], export_dir: Path = None):
    out_sig_dict = {}
    freq_gains_dict = get_settings_for_words(EQ_GAINS_PATH, word_list)
    for word_i in word_list:
        out_sig_i = apply_single_word_EQ_to_batch(signal_batch, word_i)
        print(f'applied {word_i} EQ settings to batch...')
        out_sig_dict[word_i] = out_sig_i
        if export_dir is not None:
            save_sig_batch(preprocess_audio(out_sig_i), word_i, export_dir)# test_ex_dir/f'multirun_1')

    return out_sig_dict

# applying single word EQ params (e.g. 'warm') directly on a dir of .wavs
def apply_single_word_EQ_to_dir(samples_dir: Path, word: str) -> AudioSignal:
    in_sig_batch = wav_dir_to_batch(samples_dir)
    out_sig_batch = apply_single_word_EQ_to_batch(in_sig_batch, word)
    return out_sig_batch

# applying list of words EQ params (e.g. ['warm', 'cool])directly on a dir of .wavs
def apply_multi_word_EQ_to_dir(samples_dir: Path, word_list: List[str], export_dir: Path = None) -> Dict[str, AudioSignal]:
    in_sig_batch = wav_dir_to_batch(samples_dir)
    out_sig_dict = apply_multi_word_EQ_to_batch(in_sig_batch, word_list, export_dir)
    return out_sig_dict #returns dict of d[word] = EQ'd batch_AudioSignal

# saving a batch_sig to folder of /text/.wavs 
def save_sig_batch(sig_batched, text, parent_dir_to_save_to):
    # where text = word_target
    for i, s in enumerate(sig_batched):
        instrument_dir = parent_dir_to_save_to/f'{sig_batched.path_to_file[i].stem}'
        instrument_dir.mkdir(parents=True, exist_ok=True)
        # print(instrument_path)
        s.write(instrument_dir/f"{text}.wav")
        print(f'saved {i} of batch {sig_batched.batch_size}')

