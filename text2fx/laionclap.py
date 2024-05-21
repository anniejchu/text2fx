

import os
from pathlib import Path
import requests
from tqdm import tqdm
from typing import Union, List

import math
import torch
import numpy as np
import audiotools
import dasp_pytorch
import auraloss
import laion_clap
import transformers
from audiotools import AudioSignal

from transformers import BertForMaskedLM

import matplotlib.pyplot as plt
from text2fx.core import AbstractCLAPWrapper, download_file, PRETRAINED_DIR, DEVICE


class LAIONCLAPWrapper(AbstractCLAPWrapper):
    def __init__(self):
        CLAP_MODELS = [
            '630k-best.pt', # Best non-fusion checkpoint, good for general audio < 10s
            '630k-audioset-best.pt',  # Best non-fusion checkpoint, good for general audio < 10s
            '630k-fusion-best.pt',  # Best fusion checkpoint, good for general audio of variable lengths > 10s
            '630k-audioset-fusion-best.pt',  # Best fusion checkpoint, good for general audio of variable lengths > 10s
            'music_audioset_epoch_15_esc_90.14.pt',  # Specialized for music, best music-tagging performance
            'music_speech_epoch_15_esc_89.25.pt',  # Specialized for music and speech, near-best music-tagging performance
            'music_speech_audioset_epoch_15_esc_89.98.pt',  # For music / speech / general audio, lower music-tagging performance
        ]

        # Valid CLAP audio encoder names
        CLAP_AUDIO_MODELS = [
            'HTSAT-base',
            'HTSAT-large',
            'HTSAT-tiny',  # Default
            'HTSAT-tiny-win-1536',
            'PANN-6',
            'PANN-10',
            'PANN-14',
            'PANN-14-fmax-8k-20s',
            'PANN-14-fmax-18k',
            'PANN-14-tiny-transformer',
            'PANN-14-win-1536'
        ]
        CLAP_SAMPLE_RATE = 48_000
        CLAP_PRETRAINED_DIR = PRETRAINED_DIR / "clap"
        CLAP_DOWNLOAD_LINK = 'https://huggingface.co/lukewys/laion_clap/resolve/main/'
        CLAP_MODEL_IDX = 1
        CLAP_AUDIO_MODEL_IDX = 2  # Switching this seems to cause issues... it could be all models were trained with HTSAT-tiny
        ENABLE_FUSION = CLAP_MODEL_IDX in [2, 3]  # Only some models were trained with segment fusion

        ENABLE_FUSION = False

        # Ensure that weights are downloaded
        ckpt = CLAP_MODELS[CLAP_MODEL_IDX]
        ckpt_pth = CLAP_PRETRAINED_DIR / ckpt

        if not os.path.exists(ckpt_pth):
            CLAP_PRETRAINED_DIR.mkdir(exist_ok = True, parents= True) #AC added. 
            print(f"Downloading weights for checkpoint {ckpt}")
            ckpt_pth = download_file(CLAP_DOWNLOAD_LINK + ckpt, CLAP_PRETRAINED_DIR)

        model = laion_clap.CLAP_Module(enable_fusion=ENABLE_FUSION, amodel=CLAP_AUDIO_MODELS[CLAP_AUDIO_MODEL_IDX])
        model.load_ckpt(ckpt_pth)

        model = model.to(DEVICE)

        # Ensure model does not track parameter gradients (wastes memory)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False