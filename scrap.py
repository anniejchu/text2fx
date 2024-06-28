import os
import shutil
import audiotools as at
from audiotools import AudioSignal
from pathlib import Path
import dasp_pytorch

import torch
import text2fx.core as tc
from text2fx.core import ParametricEQ_40band, Channel, functional_parametric_eq_40band
from text2fx.constants import EQ_freq_bands, EQ_words_top_10, NOTEBOOKS_DIR, SAMPLE_RATE, DEVICE

# ---- Loading EQ params
aud_csv_path_EQ = NOTEBOOKS_DIR / 'audealize_data/eqdescriptors.json'
EQ_gains_dict = tc.get_settings_for_words(aud_csv_path_EQ, EQ_words_top_10)
EQ_gains_tuple_dict = tc.convert_to_freq_gain_tuples(EQ_gains_dict, EQ_freq_bands) # refactoring to 40 (freq, gain) tuples
EQ_w_gain_tensor_settings = tc.convert_to_tensors(EQ_gains_tuple_dict) # Converting all parameters into tensors

# print(EQ_w_gain_tensor_settings['warm'])
# print(EQ_gains_dict['warm'])

# ----- Creating test instance of implemented 40band
m40b = ParametricEQ_40band(sample_rate=44100)
# print(f'num of params in 40 band (should be 40): {m40b.num_params}')

# -------- Loading all 40 gain_values for 'warm' or creating a bunch of zeros
warm_EQ_gains = torch.tensor(EQ_gains_dict['warm'])*5 #audealize implements this via *5 #torch.zeros(1, m40b.num_params).squeeze()
# print(warm_EQ_gains)
# print(f'raw warm EQ gains #: {len(warm_EQ_gains)} // values: \n{warm_EQ_gains}')

# ------- Normalizing gain values to [0, 1]
# test_params_sigmoid = torch.sigmoid(warm_EQ_gains) # diff normalization to [0, 1], don't use
warm_EQ_gains_normalized = dasp_pytorch.modules.normalize(warm_EQ_gains, m40b.min_gain_db, m40b.max_gain_db)
# print(f'post normalization warm EQ gains #: {len(warm_EQ_gains_normalized)} // values \n{warm_EQ_gains_normalized}')

# ------- Converting back to raw values
denormed_dict_sigmoided = {}
for i, param_name in enumerate(m40b.param_ranges):
    denormed_dict_sigmoided[param_name] = warm_EQ_gains_normalized[i]
init_params = m40b.denormalize_param_dict(denormed_dict_sigmoided)
# print(f'sig --> back to raw (should be same): \n{init_params}')


# --- testing funcitonal
def testing_func_basic():
    # input_tensor = torch.randn(1, 2, 44100)
    test_sig = AudioSignal('assets/multistem_examples/bass_10s.wav').to(DEVICE)
    
    q_value = 4.31
    band_gains = warm_EQ_gains #[torch.tensor(1.0)] * 40

    output_tensor = functional_parametric_eq_40band(test_sig.samples, test_sig.sample_rate, *band_gains, q=q_value)
    print(output_tensor)

# TODO: NOT WORKING, 
def testing_functional(signal): # not working
    m40b = ParametricEQ_40band(sample_rate=44100)
    channel = Channel(m40b)

    batch_size = 4  # for debugging

    # Use GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    signal = signal.to(device)
    signal = AudioSignal.batch([signal] * batch_size)

    params = torch.randn(signal.batch_size, channel.num_params).to(device)
    # params = torch.randn(1, channel.num_params).expand(signal.batch_size, -1).to(device) #random params copied across batch size

    # breakpoint()    
    # Apply effect with out estimated parameters
    signal_effected = channel(signal, torch.sigmoid(params)).clone().detach().cpu()

    return signal_effected

def save_sig_batch(sig_batched, save_dir):
    for i, s in enumerate(sig_batched):
        print(i, s)
        s.write(save_dir/f"{i}_final.wav")

if __name__ == "__main__":

    # testing_func_basic() # working

    test_sig = AudioSignal('assets/multistem_examples/bass_10s.wav').to(DEVICE)
    testing_functional(test_sig)
    # _m40b = ParametricEQ_40band(sample_rate=44100)
    # channel = Channel(_m40b)
    # batch_size = 4  # for debugging
    # signal = AudioSignal.batch([test_sig] * batch_size)
    # print(signal.samples)

    # # testing wrapped function
    # y_wrap = testing_functional(test_sig, batch_size, channel)
    # # test_ex_dir = Path('experiments/paramEQ_40')
    # # save_sig_batch(y, test_ex_dir)

    # # testing function directly
    # y_func = testing_func_basic(test_sig, test_sig.sample_rate)


### old
# top10_eq = EQ_words_top_10

# def rename_seed_dirs(root_dir):
#     for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
#         for dirname in dirnames:
#             if dirname.startswith('seed_') and dirname.endswith('_0'):
#                 new_dirname = dirname.replace('seed_', 'seed')
#                 old_path = os.path.join(dirpath, dirname)
#                 new_path = os.path.join(dirpath, new_dirname)
#                 os.rename(old_path, new_path)
#                 print(f'Renamed: {old_path} -> {new_path}')

# def group_subdirectories(parent_dir):
#     categories = {
#         'bright': 'normal',
#         'warm': 'normal',
#         'heavy': 'normal',
#         'soft': 'normal',
#         'brighter': 'comparatives',
#         'warmer': 'comparatives',
#         'heavier': 'comparatives',
#         'softer': 'comparatives',
#         'very_bright': 'emphasis',
#         'very_warm': 'emphasis',
#         'very_heavy': 'emphasis',
#         'very_soft': 'emphasis',
#         # Uncomment if 'less' categories are also present
#         'less_bright': 'subdue',
#         'less_warm': 'subdue',
#         'less_heavy': 'subdue',
#         'less_soft': 'subdue'
#     }
    
#     for subdir in os.listdir(parent_dir):
#         subdir_path = os.path.join(parent_dir, subdir)
#         if os.path.isdir(subdir_path) and subdir in categories:
#             category = categories[subdir]
#             target_dir = os.path.join(parent_dir, category)
#             if not os.path.exists(target_dir):
#                 os.makedirs(target_dir)
#             shutil.move(subdir_path, target_dir)

# def cut_to_10s_file(wav2cut_path, output_path):
#     cut_sig = AudioSignal(wav2cut_path, duration=10)
#     out_write = f'{output_path}/{wav2cut_path.stem}_10s.wav'
#     cut_sig.write(out_write)

# def cut_to_10s_dir(wavs2cut_dir_path, output_dir):
#     for file in os.listdir(wavs2cut_dir_path):
#         full_file_path = wavs2cut_dir_path/file
#         cut_to_10s_file(full_file_path, output_dir)

# def delete_specific_files(parent_dir, filenames):
#     """
#     Searches for and deletes specific files within a given parent directory.

#     :param parent_dir: The parent directory to search within.
#     :param filenames: A list of filenames to search for and delete.
#     """
#     for root, dirs, files in os.walk(parent_dir):
#         for file in files:
#             if file in filenames:
#                 file_path = os.path.join(root, file)
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted: {file_path}")
#                 except OSError as e:
#                     print(f"Error deleting {file_path}: {e}")

# def cut_to_10s_file_overwrite(wav2cut_path):
#     """
#     Cuts a single WAV file to 10 seconds and overwrites the original file.
    
#     Parameters:
#     - wav2cut_path (str): Path to the WAV file to cut.
#     """
#     try:
#         cut_sig = AudioSignal(wav2cut_path, duration=10)
#         cut_sig.write(wav2cut_path)  # Overwrite the original file
#         print(f"Processed: {wav2cut_path}")
#     except Exception as e:
#         print(f"Error processing {wav2cut_path}: {e}")

# def cut_to_10s_dir_overwrite(wavs2cut_dir_path):
#     """
#     Recursively cuts all WAV files in a directory to 10 seconds and overwrites the original files.
    
#     Parameters:
#     - wavs2cut_dir_path (str): Directory containing WAV files to cut.
#     """
#     for root, dirs, files in os.walk(wavs2cut_dir_path):
#         for file in files:
#             if file.endswith('.wav'):
#                 full_file_path = os.path.join(root, file)
#                 cut_to_10s_file_overwrite(full_file_path)

# def delete_specific_directories(parent_dir, directory_names):
#     """
#     Searches for and deletes directories within a given parent directory based on exact names.

#     :param parent_dir: The parent directory to search within.
#     :param directory_names: A list of directory names to search for and delete.
#     """
#     [shutil.rmtree(os.path.join(root, dir_name)) for root, dirs, _ in os.walk(parent_dir, topdown=False)
#      for dir_name in dirs if dir_name in directory_names]

# if __name__ == "__main__":

    # delete_specific_files(parent_directory, files_to_delete)
#     # parent_directory = 'runs/test625_wordtargets/laion_clap/drums_beatles_musdb'
#     # group_subdirectories(parent_directory)
#     to_cut_dir = Path('assets/multistem_examples/full_samples')
#     cut_export_dir = 'assets/multistem_examples'
#     single_file = Path('assets/audealize_examples/piano.wav')
#     cut_to_10s_file(single_file, cut_export_dir)

    # # Usage
    # parent_directory = Path('experiments/multistem_test')
    # cut_to_10s_dir_overwrite(parent_directory)

    
