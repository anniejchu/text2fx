import os
import shutil
import audiotools as at
from audiotools import AudioSignal
from pathlib import Path
import dasp_pytorch
from typing import Union, List, Optional, Tuple, Iterable, Dict


import torch
import text2fx.core as tc
from text2fx.core import ParametricEQ_40band, Channel, functional_parametric_eq_40band
from text2fx.constants import EQ_freq_bands, EQ_words_top_10, NOTEBOOKS_DIR, SAMPLE_RATE, DEVICE, EQ_GAINS_PATH
from text2fx.__main__ import text2fx

# --- testing funcitonal
def testing_func_basic():
    # input_tensor = torch.randn(1, 2, 44100)
    test_sig = AudioSignal('assets/multistem_examples/10s/bass.wav').to(DEVICE)
    
    q_value = 4.31
    band_gains = [torch.tensor(1.0)] * 40 #warm_EQ_gains

    output_tensor = functional_parametric_eq_40band(test_sig.samples, test_sig.sample_rate, *band_gains, q=q_value)
    print(output_tensor)

# apply Audealize EQ settings for a single word on a single sig #testing batch
def apply_single_word_EQ_to_sig(sig_single: AudioSignal, batch_size: int,freq_gains_dict: dict, word:str='none'):
    m40b = ParametricEQ_40band(sample_rate=44100)
    channel = Channel(m40b)

    # Use GPU if available
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    signal = sig_single.to(device)
    signal = AudioSignal.batch([signal] * batch_size)

    if word=='none':
        print(f'... setting random params')
        params = torch.randn(signal.batch_size, channel.num_params).to(device)
    else:
        print(f'...getting parameters for {word}')
        param_single = torch.tensor(freq_gains_dict[word])*5 #make dict parameter
        params = param_single.expand(signal.batch_size, -1).to(device)
    signal_effected_batch = channel(signal, torch.sigmoid(params)).clone().detach().cpu()

    return signal_effected_batch

# testing CLAP optimization
def test_text2fx_batch(channel, in_sig_batch: AudioSignal, model_name: str, test_name: str):
    all_out=[]
    # save_text = "test5_this_is_a_X_sound_random"
    save_dir = tc.create_save_dir(test_name, Path("experiments"))
    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in ["warm", "bright", "deep"]:
        # Apply text2fx
            signal_effected = text2fx(
                model_name=model_name, 
                sig=in_sig_batch, 
                text=text_target, 
                channel=channel,
                criterion=criterion, 
                save_dir=save_dir / text_target / criterion,
                params_init_type='random',
                seed_i=3,
                roll='all',
                # roll_amt=10000
            )
            all_out.append(signal_effected)
    return all_out

if __name__ == "__main__":
    # m40b = ParametricEQ_40band(sample_rate=SAMPLE_RATE)
    # channel_40_band = Channel(
    #     m40b
    #     )
    # print(channel_40_band.modules)
    # input_samples_dir = Path('assets/multistem_examples/10s')
    # in_sig_batched = wav_dir_to_batch(input_samples_dir)
    # print(in_sig_batched.batch_size)

    # run_name = "this_is_a_X_sound_random"
    # out = test_text2fx_batch(channel_40_band, in_sig_batched, 'ms_clap', test_name=run_name)
    # print(out)
    #--- Testing other
    #constants
    test_ex_dir = Path('experiments/2024-07-03/OTHER')
    input_samples_dir = Path('assets/multistem_examples/10s')

    #vary
    w_list = ['crunch', 'dramatic', 'muffled']
    w_list_out = tc.apply_multi_word_EQ_to_dir(input_samples_dir, w_list, export_dir=test_ex_dir)
    print(w_list_out)
    
    freq_gains_dict = tc.get_settings_for_words(EQ_GAINS_PATH, w_list[0])
    print(freq_gains_dict)
    # test_ex_dir = Path('experiments/paramEQ_40')
    # input_samples_dir = Path('assets/multistem_examples/10s')
    # aud_csv_path_EQ = NOTEBOOKS_DIR / 'audealize_data/eqdescriptors.json'
    # EQ_gains_dict = tc.get_settings_for_words(aud_csv_path_EQ, EQ_words_top_10)

    # # other_EQ_gains_dict = tc.get_settings_for_words(aud_csv_path_EQ, EQ_words_other)

    # for word_t in EQ_gains_dict.keys():
    #     out_sig_b = single_word_on_batch_sig(input_samples_dir, EQ_gains_dict, word_t)
    #     save_sig_batch(tc.preprocess_sig(out_sig_b), word_t, test_ex_dir/f'multirun_1')
    # # print(other_EQ_gains_dict)
