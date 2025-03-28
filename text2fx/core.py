from pathlib import Path
import datetime
import unicodedata
import re
import random
import json
import os
from typing import Union, List, Optional, Tuple, Iterable, Dict

import torch
import torchaudio.transforms as T
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm

import audiotools as at
from audiotools import AudioSignal
import dasp_pytorch
import auraloss
from functools import partial
from collections import defaultdict
import numbers

from text2fx.constants import EQ_freq_bands, SAMPLE_RATE, EQ_GAINS_PATH, DEVICE

def norm(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

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

            _params_normalized = torch.sigmoid(_params)

            raw_param_dict = m.extract_param_dict(_params_normalized)
            # breakpoint()
            denorm_param_dict = m.denormalize_param_dict(raw_param_dict)

            # denorm_param_dict = {k: v.tolist() for k, v in denorm_param_dict.items()}
            all_params[m.__class__.__name__] = denorm_param_dict

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



# def normalize_param_dict(param_dict: dict, fx):
#     """Given parameters on (0,1) restore them to the ranges expected by the processor.

#     Args:
#         param_dict (dict): Dictionary of parameter tensors on (0,1).

#     Returns:
#         dict: Dictionary of parameter tensors on their full range.

#     """
#     all_params = {}
#     params_count = 0
#     denorm_param_dict = {}
#     for param_name, param_tensor in param_dict['ParametricEQ'].items():
#         # print(param_name, param_tensor)
#         param_val_denorm = normalize(
#             param_tensor[0],
#             fx.param_ranges[param_name][0],
#             fx.param_ranges[param_name][1],
#         )
#         denorm_param_dict[param_name] = param_val_denorm
#     return denorm_param_dict
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
    for i in range(len(EQ_freq_bands)):
        band_gain_i = locals()[f"band{i}_gain_db"] 

        # Design peak filter
        b, a = dasp_pytorch.signal.biquad(band_gain_i, band_freqs[:, i], qs, sample_rate, 'peaking')
        
        x_out = dasp_pytorch.signal.lfilter_via_fsm(x_out, b, a)

    x_out= x_out.view(nb,nc,nt) #this should be output

    return x_out # Returns: x_out (torch.Tensor): filtered signal


def apply_EQ_file(file_name, freqs, Q=4.31): #process function
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

def apply_audealize_single_word(input_audio_file: AudioSignal, text: Union[str, List[str]], save_dir: Union[str, Path]):
    m40b = ParametricEQ_40band(sample_rate=44100)
    channel = Channel(m40b)
    if isinstance(text, str):
        text = [text]    

    freq_gains_dict = get_settings_for_words(EQ_GAINS_PATH, text)
    # Use GPU if available

    signal = AudioSignal(input_audio_file).to(DEVICE)

    for t in text:
        param_single = torch.tensor(freq_gains_dict[t])*5 #make dict parameter
            # print(f'got it: {freq_gains_dict[word]}')
        params = param_single.expand(signal.batch_size, -1).to(DEVICE)

        signal_effected= channel(signal, torch.sigmoid(params)).clone().detach().cpu()
        export_sig(signal_effected, save_dir)
        
    return signal_effected

# def load_examples(dir_path: Union[str,Path]) -> List[Path]:
#     dir_path = Path(dir_path)  # Convert string to Path if necessary
#     exts = ["mp3", "wav", "flac"]
#     example_files = [list(dir_path.rglob(f"*.{e}")) for e in exts]
#     example_files = sum(example_files, [])  # Trick to flatten list of lists
#     return example_files

def load_examples(dir_path: Union[str, Path]) -> List[Path]:
    dir_path = Path(dir_path)  # Convert string to Path if necessary
    
    if dir_path.is_file():  # If it's a single file, return it as a list
        return [dir_path]
    
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
    gains = np.array(gains)
    
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

def preprocess_audio(audio_path_or_array: Union[torch.Tensor, str, Path, np.ndarray, AudioSignal], 
                     salient_excerpt_duration: Optional[int] = None, 
                     sample_rate: Optional[int] = None) -> AudioSignal:
    """Preprocesses an audio input (file path, tensor, ndarray, or AudioSignal).
    
    Args:
        audio_path_or_array: The audio input, can be a file path, tensor, ndarray, or AudioSignal.
        salient_excerpt_duration: If provided, extracts the salient excerpt of this duration.
        sample_rate: Required if input is a tensor or ndarray.
        
    Returns:
        Processed `AudioSignal`.
    """

    if isinstance(audio_path_or_array, (str, Path)):  
        sig = AudioSignal(audio_path_or_array)  
    elif isinstance(audio_path_or_array, AudioSignal):
        sig = audio_path_or_array  
    elif isinstance(audio_path_or_array, (torch.Tensor, np.ndarray)):
        if sample_rate is None:
            raise ValueError("Must provide `sample_rate` if input is a tensor or ndarray")
        sig = AudioSignal(audio_path_or_array, sample_rate)
    else:
        raise ValueError("Input must be a file path, AudioSignal, tensor, or ndarray")
    
    # Standard processing: convert to mono, resample, ensure max normalization
    sig = sig.to_mono().resample(SAMPLE_RATE).ensure_max_of_audio()
    
    # Apply salient excerpt if specified
    if salient_excerpt_duration:
        return at_salient_excerpt(sig, duration=salient_excerpt_duration, loudness_cutoff=0)

    return sig

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


def printy(path_dir: Union[Path, List[Path]]):
    if isinstance(path_dir, list):
        for path in path_dir:
            print(path)
    else:
        print(path_dir)


def wavs_to_batch(samples: Union[str, Path, List[str], List[Path]]) -> AudioSignal:
    if isinstance(samples, (str, Path)):
        all_raw_sigs = load_examples(samples)
    else:
        samples = [Path(s) if isinstance(s, str) else s for s in samples]
        all_raw_sigs = samples
    
    signal_list = [preprocess_audio(raw_sig_i) for raw_sig_i in all_raw_sigs]
    return AudioSignal.batch(signal_list, pad_signals=True)


def create_channel(fx_chain, sr=SAMPLE_RATE):
    module_map = {
        'gain': dasp_pytorch.Gain,
        'distortion': Distortion,
        'parametriceq': dasp_pytorch.ParametricEQ,
        'parametric equalizer': dasp_pytorch.ParametricEQ,
        'eq': dasp_pytorch.ParametricEQ,
        'dynamic range compressor': dasp_pytorch.Compressor,
        'reverb': dasp_pytorch.NoiseShapedReverb,
        'noiseshapedreverb': dasp_pytorch.NoiseShapedReverb,
        'compressor': dasp_pytorch.Compressor,
        'compression': dasp_pytorch.Compressor,
        'eq40': ParametricEQ_40band,
    }

    modules = []
    for fx in fx_chain:
        fx_name = fx.lower()
        if fx_name in module_map:
            modules.append(module_map[fx_name](sample_rate=sr))
        else:
            raise ValueError(f"Unsupported FX: {fx}")

    return Channel(*modules)

def export_sig(out_sig: AudioSignal, save_path_or_dir: Union[str, Path], text: Union[str, List[str]] = None):
    # Ensure the directory exists
    save_path_or_dir = Path(save_path_or_dir)

    if out_sig.batch_size == 1:
        save_path_or_dir.parent.mkdir(parents=True, exist_ok=True)
        assert not save_path_or_dir.is_dir(), "save_path_or_dir must be a filename, not a directory, when batch_size == 1"
        out_sig.clone().detach().cpu().write(save_path_or_dir)
    else:
        save_path_or_dir.mkdir(parents=True, exist_ok=True)
        assert save_path_or_dir.is_dir(), "save_path_or_dir MUST be a directory when batch_size > 1"
        for i, s in enumerate(out_sig):
            if text:
                save_path = save_path_or_dir/f'{i}_{text[i]}_{out_sig.path_to_file[i].stem}.wav'
            else:
                save_path = save_path_or_dir/f'{i}_{out_sig.path_to_file[i].stem}.wav'
            s.write(save_path)
            print(f'saved {i+1} of batch {out_sig.batch_size}')

def detensor_dict(input_dict: dict) -> dict:
    output_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else
        {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in value.items()} if isinstance(value, dict) else value for key, value in input_dict.items()}
    return output_dict

def flatten_single_item_lists(d):
    for key, subdict in d.items():
        for subkey, value in subdict.items():
            if isinstance(value, list) and len(value) == 1:
                subdict[subkey] = value[0]
    return d

def save_dict_to_json(params_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json_serializable_dict = flatten_single_item_lists(detensor_dict(params_dict))
    # Save the dictionary to JSON file
    with open(save_path, 'w') as f:
        json.dump(json_serializable_dict, f, indent=4)

def save_params_batch_to_jsons(in_dict, save_dir, data_labels: List[Tuple[Path, str]]=None): #out_sig_to_match: AudioSignal = None, text_to_match: List[str] = None):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    input_dict = detensor_dict(in_dict)

    # Initialize a defaultdict to collect data by index
    index_data = defaultdict(dict)
    for module, params in input_dict.items():
        for param, values in params.items():
            for idx, value in enumerate(values):
                index_data[idx].setdefault(module, {})[param] = value #to make flat and not list of scalar value, remove brackets

    # Save each index data as a separate JSON file using save_dict_to_json
    for idx, data in index_data.items():
        print(f'splitting json with {len(index_data)} audio/text pairs ... {idx+1} / {len(index_data)}')
        if data_labels:
            assert len(index_data) == len(data_labels), "Number of indices in input_dict must match out_sig_to_match.batch_size"
            file_path = os.path.join(save_dir, f"{idx}_{data_labels[idx][1]}_{data_labels[idx][0].stem}.json")
        else:
            file_path = os.path.join(save_dir, f"{idx}_index.json")
        save_dict_to_json(data, file_path)


def load_words(words_source: Union[str, Path, List[str]]) -> List[str]:
    """
    Samples n words from the given word descriptor source.

    :param words_source: File containing word descriptors (one per line) or a list of descriptors.
    :return: List of sampled descriptor words.
    """
    if isinstance(words_source, (str, Path)):
        # Check if the words_source is a single word (not a file path)
        if len(words_source.split()) == 1:
            word_list = [words_source.strip()]
        else:
            with open(words_source, 'r') as f:
                word_list = [line.strip() for line in f if line.strip()]
    else:
        word_list = words_source

    return word_list


#hacking audiotools salient excerpt to work on AudioSignal type 
def random_state(seed: Union[int, np.random.RandomState]):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : typing.Union[int, np.random.RandomState] or None
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState
        Random state object.

    Raises
    ------
    ValueError
        If seed is not valid, an error is thrown.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
        )

def at_excerpt(signal: AudioSignal,
            offset: float = None,
            duration: float = None,
            state: Union[np.random.RandomState, int] = None):
    signal = signal.clone()
    total_duration = signal.duration
    state = random_state(state)
    lower_bound = 0 if offset is None else offset
    upper_bound = max(total_duration - duration, 0)
    offset = state.uniform(lower_bound, upper_bound)

        # Convert offset and duration to number of samples
    offset_samples = int(offset * signal.sample_rate)
    duration_samples = int(duration * signal.sample_rate)
    signal.audio_data = signal.audio_data[..., offset_samples:offset_samples + duration_samples]

    signal.metadata["offset"] = offset
    signal.metadata["duration"] = duration
    return signal

def at_salient_excerpt(
        sig: AudioSignal,
        duration: int, 
        loudness_cutoff: float = None,
        num_tries: int = 8,
        state: Union[np.random.RandomState, int] = None):

    state = random_state(state)
    if loudness_cutoff is None:
        excerpt = at_excerpt(sig, duration=duration, state=state)
    else:
        loudness = -np.inf
        num_try = 0
        while loudness <= loudness_cutoff:
            excerpt = at_excerpt(sig, duration=duration, state=state)
            loudness = excerpt.loudness()
            num_try += 1
            if num_tries is not None and num_try >= num_tries:
                break
    return excerpt
