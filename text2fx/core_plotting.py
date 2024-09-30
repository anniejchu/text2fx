import os
import sys
import warnings
from contextlib import contextmanager

from pathlib import Path
import random
import numpy as np
import torch
import math
import json

from audiotools import AudioSignal
from typing import Union, List, Optional, Tuple, Iterable, Dict, Union
import dasp_pytorch
import matplotlib.pyplot as plt



#################### plot frequency response on 2 audio files (audiosignal) ##############
def plot_response(
    y: AudioSignal,
    y_hat: AudioSignal,
    sample_rate: int = 44100,
    tag: str = None,
    save_path = None):
    fig, ax = plt.subplots(figsize=(6, 4))

    y = y.resample(sample_rate)
    y_hat = y_hat.resample(sample_rate)
    
    y = y.to_mono().audio_data.flatten().cpu().detach()
    y_hat = y_hat.to_mono().audio_data.flatten().cpu().detach()

    # compute frequency response of y
    Y = torch.fft.rfft(y)
    Y = torch.abs(Y)
    Y_db = 20 * torch.log10(Y + 1e-8)

    # compute frequency response of x_hat
    Y_hat = torch.fft.rfft(y_hat)
    Y_hat = torch.abs(Y_hat)
    Y_hat_db = 20 * torch.log10(Y_hat + 1e-8)

    # compute frequency axis
    freqs = torch.fft.fftfreq(y.shape[-1], d=1 / sample_rate)
    freqs = freqs[: Y.shape[-1] - 1]  # take only positive frequencies
    Y_db = Y_db[: Y.shape[-1] - 1]
    Y_hat_db = Y_hat_db[: Y_hat.shape[-1] - 1]
    
    # smooth frequency response
    kernel_size = 1023
    Y_hat_db = torch.nn.functional.avg_pool1d(
        Y_hat_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    Y_db = torch.nn.functional.avg_pool1d(
        Y_db.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )

    # plot frequency response
    ax.plot(freqs, Y_db.squeeze().cpu().numpy(), label="Original",color='gray',linewidth=2, alpha=1)
    ax.plot(freqs, Y_hat_db.cpu().squeeze().numpy(), label=f"Effected",linewidth=2, alpha=1)

    if tag:
        ax.set_title(f'{tag}')
    # ax.set_xlabel("Frequency (Hz)")
    # ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(100, 20000)
    # ax.set_ylim(-40, 40)
    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Magnitude (dB)", fontsize=12, fontweight='bold')

    ax.set_xscale("log")
    plt.legend()
    # plt.grid(c="lightgray")
    plt.tight_layout()
    plt.show()

    if save_path:
        # save_path = save_dir/f'{tag}.png'
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {save_path}")

def plot_response_files(file1, file2, tag =''):
    sig1 = AudioSignal(file1).to_mono().resample(44_100).normalize(-24)
    sig2 = AudioSignal(file2).to_mono().resample(44_100).normalize(-24)
    plot_response(sig1, sig2, tag = tag)

def plot_all_iters(sample_dir):
    graph_dir = Path(sample_dir)
    ORIGINAL_wav = next(file for file in graph_dir.rglob('*') if 'ref' in file.name)
    prefix, _ = ORIGINAL_wav.stem.split('__', 1)
    for n in range(0, 650, 100):
        n_wav = graph_dir / f'{prefix}_{n}.wav'
        plot_response_files(ORIGINAL_wav, n_wav, tag=f'iter {n}')
    
#################### plot eq curve givne params ##############
def load_eq_params(json_data):
    """Load parametric EQ parameters from a JSON string."""
    param_eq = json_data["params"]["ParametricEQ"]
    cutoffs = [
        param_eq["low_shelf_cutoff_freq"][0],
        param_eq["band0_cutoff_freq"][0],
        param_eq["band1_cutoff_freq"][0],
        param_eq["band2_cutoff_freq"][0],
        param_eq["band3_cutoff_freq"][0],
        param_eq["high_shelf_cutoff_freq"][0],
    ]
    gains = [
        param_eq["low_shelf_gain_db"][0],
        param_eq["band0_gain_db"][0],
        param_eq["band1_gain_db"][0],
        param_eq["band2_gain_db"][0],
        param_eq["band3_gain_db"][0],
        param_eq["high_shelf_gain_db"][0],
    ]
    q_factors = [
        param_eq["low_shelf_q_factor"][0],
        param_eq["band0_q_factor"][0],
        param_eq["band1_q_factor"][0],
        param_eq["band2_q_factor"][0],
        param_eq["band3_q_factor"][0],
        param_eq["high_shelf_q_factor"][0],
    ]
    # Combine cutoffs, gains, and q_factors into a list of tuples
    return cutoffs, gains, q_factors

# Biquad filter implementation in PyTorch
def biquad(gain_db, cutoff_freq, q_factor, sample_rate, filter_type):
    bs = gain_db.size(0)
    gain_db = gain_db.view(bs, -1)
    cutoff_freq = cutoff_freq.view(bs, -1)
    q_factor = q_factor.view(bs, -1)

    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = torch.stack([b0, b1, b2], dim=1).view(bs, -1)
    a = torch.stack([a0, a1, a2], dim=1).view(bs, -1)

    # normalize
    b = b.type_as(gain_db) / a0
    a = a.type_as(gain_db) / a0

    return b, a

# Compute frequency response from biquad coefficients
def frequency_response(b, a, frequencies, sample_rate):
    w = 2 * np.pi * frequencies / sample_rate
    z = np.exp(1j * w)
    numerator = b[0] + b[1] / z + b[2] / (z**2)
    denominator = a[0] + a[1] / z + a[2] / (z**2)
    return numerator / denominator

# Convert PyTorch tensors to numpy arrays
def biquad_to_numpy(gain_db, cutoff_freq, q_factor, sample_rate, filter_type):
    gain_db = torch.tensor(gain_db, dtype=torch.float32)
    cutoff_freq = torch.tensor(cutoff_freq, dtype=torch.float32)
    q_factor = torch.tensor(q_factor, dtype=torch.float32)
    
    b, a = biquad(gain_db, cutoff_freq, q_factor, sample_rate, filter_type)
    
    return b.numpy(), a.numpy()

# Create a frequency response function using biquads
def parametric_eq_response_biquad(frequencies, gains_db, cutoffs, q_factors, sample_rate=44100):
    response = np.ones_like(frequencies, dtype=complex)
    
    for i, (gain, cutoff, q) in enumerate(zip(gains_db, cutoffs, q_factors)):
        filter_type = "peaking" if i not in [0, 5] else ("low_shelf" if i == 0 else "high_shelf")
        b, a = biquad_to_numpy([gain], [cutoff], [q], sample_rate, filter_type)
        band_response = frequency_response(b[0], a[0], frequencies, sample_rate)
        response *= band_response  # Apply each filter in series
    
    response_db = 20 * np.log10(np.abs(response))  # Convert to dB scale
    response_db -= np.max(response_db)/8     # Normalize dB
    return response_db

def plot_parametric_eq(j, label='', save_dir=None):
    init_cutoffs, init_gains_db, init_q_factors = load_eq_params(j)
    frequencies = np.logspace(1, 4.3, 500)  # Log-spaced frequencies from 10 Hz to 20 kHz
    fig, ax = plt.subplots(figsize=(10,4), facecolor='#ffffff')
    plt.subplots_adjust(left=0.25, right=0.75, top=0.75, bottom=0.25)

    # Plot the initial frequency response using biquads
    response = parametric_eq_response_biquad(frequencies, init_gains_db, init_cutoffs, init_q_factors)
    eq_line, = ax.plot(frequencies, response, lw=2, color='black')

    init_cutoffs = np.array(init_cutoffs)
    cutoff_responses = parametric_eq_response_biquad(init_cutoffs, init_gains_db, init_cutoffs, init_q_factors)
    ax.scatter(init_cutoffs, cutoff_responses, color = 'black', marker='o', s=80, zorder=5, alpha=1, linewidths=2, edgecolors='red',label='Frequencies')

    ax.set_xscale('log')
    ax.set_xlim(100, 20000)
    ax.set_ylim([-40, 40])
    ax.set_title(f'({label}) 6-Band Parametric EQ Interface', color='black', fontweight='bold', pad=10)
    ax.set_xlabel('Frequency (Hz)', color='black', fontweight='bold')
    ax.set_ylabel('Gain (dB)',color='black', fontweight='bold')
    ax.grid(False)
    ax.grid(which='major', color='black', linewidth=0.3)
    ax.grid(which='minor', color='black', linestyle=':', linewidth=0.5, alpha=0.6)
    [ax.spines[side].set_visible(False) for side in ax.spines]
    ax.tick_params(color='darkgray', labelcolor='darkgray')
    # ax.set_facecolor('#525252')

    if save_dir:
        save_path = save_dir/f'{label}.png'
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {save_path}")

def plot_parametric_eq_from_json(json_file_path, export_dir=None, do_all=True, n_iter=0):
    with open(json_file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        filtered_data = [entry for entry in data if entry.get('iteration') == n_iter]
    if do_all:
        # out_dir = json_file_path.parent/'justgraph'
        if export_dir:
            export_dir.mkdir(parents=True, exist_ok=True)
        for i in data:
            plot_parametric_eq(i, f"iter {i['iteration']}", save_dir=export_dir)
    else:
        plot_parametric_eq(filtered_data[0])

def inspect_params(json_file_path, do_all=True, n_iter=200):
    with open(json_file_path, 'r') as file:
        data = [json.loads(line) for line in file]
        filtered_data = [entry for entry in data if entry.get('iteration') == n_iter]
    return data if do_all else filtered_data[0]