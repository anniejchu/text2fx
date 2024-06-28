import os
import shutil
import audiotools as at
from audiotools import AudioSignal
from pathlib import Path
import dasp_pytorch

import torch
import text2fx.core as tc
from text2fx.core import ParametricEQ_40band, Channel
from text2fx.constants import EQ_freq_bands, EQ_words_top_10, NOTEBOOKS_DIR, SAMPLE_RATE, DEVICE

# Loading EQ params
aud_csv_path_EQ = NOTEBOOKS_DIR / 'audealize_data/eqdescriptors.json'
EQ_gains_dict = tc.get_settings_for_words(aud_csv_path_EQ, EQ_words_top_10)
EQ_gains_tuple_dict = tc.convert_to_freq_gain_tuples(EQ_gains_dict, EQ_freq_bands) # refactoring to 40 (freq, gain) tuples
EQ_w_gain_tensor_settings = tc.convert_to_tensors(EQ_gains_tuple_dict) # Converting all parameters into tensors
# print(EQ_gains_dict['warm'])

# Creating test instance of implemented 40band
m40b = ParametricEQ_40band(sample_rate=44100)
print(f'num of params in 40 band: should be 40: {m40b.num_params}')

# Loading gain_values for 'warm' (40 values) or creating a bunch of zeros
zeros_EQ_gains = torch.zeros(1, m40b.num_params).squeeze()
warm_EQ_gains = torch.tensor(EQ_gains_dict['warm'])*5 #audealize implements this via *5
# print(warm_EQ_gains)
print(f'raw warm EQ gains #: {len(warm_EQ_gains)} // values: \n{warm_EQ_gains}')

# Putting gain values through sigmoid ... / normalize
test_params_sigmoid = torch.sigmoid(warm_EQ_gains)
test_params_normalize = dasp_pytorch.modules.normalize(warm_EQ_gains, -20, 20)
# test_params_normalize = (warm_EQ_gains - warm_EQ_gains.min()) / (warm_EQ_gains.max() - warm_EQ_gains.min())
# print(test_params_sigmoid)
print(f'post normalization warm EQ gains #: {len(test_params_normalize)} // values \n{test_params_normalize}')

# Converting back to raw values
denormed_dict_sigmoided = {}
for i, param_name in enumerate(m40b.param_ranges):
    denormed_dict_sigmoided[param_name] = test_params_normalize[i]
init_params = m40b.denormalize_param_dict(denormed_dict_sigmoided)
print(f'sig --> back to raw: \n{init_params}')

# m1 = dasp_pytorch.ParametricEQ(sample_rate=44100)
# print(m1.num_params)
# print(len(warm_EQ_gains))
# m1_40band_EQ = 

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

    
