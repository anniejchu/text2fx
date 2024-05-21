from pathlib import Path
from tqdm import tqdm
import datetime
import unicodedata
import re

import torch
import torchaudio.transforms as T
import numpy as np
import audiotools as at
import dasp_pytorch
from audiotools import AudioSignal
from typing import Iterable
import random
from torch.utils.tensorboard import SummaryWriter

from msclap import CLAP

from text2fx.core import SAMPLE_RATE, Channel, AbstractCLAPWrapper, Distortion, load_audio_examples, DEVICE, create_save_dir, RUNS_DIR

import matplotlib.pyplot as plt

"""
EX CLI USAGE
python text2fx.py --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
                 --text "this sound is happy" \
                 --criterion "cosine-sim" \
                 --n_iters 600 \
                 --lr 0.01 
"""
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


def get_model(model_choice: str):
    if model_choice=="laion_clap":
        from text2fx.laionclap import LAIONCLAPWrapper
        model = LAIONCLAPWrapper()
    elif model_choice == "ms_clap":
        from text2fx.msclap import MSCLAPWrapper
        model = MSCLAPWrapper()
    else:
        raise ValueError('choose a model1!!!!!!')
    return model


def clip_directional_loss(
        a1: torch.Tensor, 
        a2: torch.Tensor, 
        b1: torch.Tensor, 
        b2: torch.Tensor
    ):
        a_dir = a1 - a2
        a_dir /= a_dir.clone().norm(dim=-1, keepdim=True)

        b_dir = b1 - b2
        b_dir /= b_dir.clone().norm(dim=-1, keepdim=True)

        loss = 1 - torch.cosine_similarity(a_dir, b_dir, dim=-1)
        return loss

def get_default_channel():
    return Channel(
        # Apply random EQ > Compression > Gain > Reverb to a signal
        dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
        dasp_pytorch.Compressor(sample_rate=SAMPLE_RATE),
        dasp_pytorch.Gain(sample_rate=SAMPLE_RATE),
        dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE),
        
        # Apply random Reverb and Distortion to a signal
        # Distortion(sample_rate=SAMPLE_RATE),
    )

def text2fx(
    sig: AudioSignal, 
    text: str,   
    channel: Channel,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    log_audio_every_n: int = 25, 
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    save_dir: str = None # figure out a save path automatically
):
    # ah yes, the max morrison trick of hiding global variables as function members
    # prevents loading the model everytime w/o needing to set it first as global variable
    if not hasattr(text2fx, "clap"): #any clap
        # msclap  = MSCLAPWrapper()
        clap = get_model("ms_clap")
        setattr(text2fx, "clap", clap)
    else: 
        clap = text2fx.clap

    # a save dir for our goods
    if save_dir is None:
        save_dir = create_save_dir(text, sig, RUNS_DIR)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # create a writer for saving stuff to tensorboard
    writer_dir = save_dir / "logs"
    writer_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(writer_dir) #SummaryWriter is tensorboard writer

    # params!
    # NOTE: these aren't actually initialized to "zeros" since the we'll apply a sigmoid which will shift this up right? 
    params = torch.nn.parameter.Parameter(
        torch.zeros(sig.batch_size, channel.num_params).to(device) 
    )
    params.requires_grad=True
    # the optimizer!
    optimizer = torch.optim.Adam([params], lr=lr)

    # log what our initial effect sounds like (w/ random parameters applied)
    init_sig = channel(sig.clone().to(device), torch.sigmoid(params))
    if writer:
        writer.add_audio("input", sig.samples[0][0], 0, sample_rate=sig.sample_rate)
        writer.add_audio("effected", init_sig.samples[0][0], 0, sample_rate=init_sig.sample_rate)

    sig.clone().cpu().write(save_dir / 'input.wav')

    embedding_target = clap.get_text_embeddings([text]).detach()
    
    if criterion == "directional_loss":
        audio_in_emb = clap.preprocess_and_embed(sig.to(device)).detach()
        text_anchor_emb = clap.get_text_embeddings(["a sound"]).detach()

    # Optimize our parameters by matching effected audio against the target audio
    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        
        # Apply effect with out estimated parameters
        signal_effected = channel(sig.to(device), torch.sigmoid(params.to(device)))

        # Get CLAP embedding for effected audio
        embedding_effected = clap.preprocess_and_embed(signal_effected) #.get_audio_embeddings takes in preprocessed audio

        # loss
        if criterion == "directional_loss":
            loss = clip_directional_loss(embedding_effected, audio_in_emb, embedding_target, text_anchor_emb).sum()
        elif criterion == "standard": #is neg dot product loss aims to minimize the dot prod b/w dissimilar items, no direction intake
            loss = -(embedding_effected @ embedding_target.T).sum()
        elif criterion == "cosine-sim": # cosine_sim loss aims to maximize the cosine similarity between similar items, normalized
            loss = 1 - torch.cosine_similarity(embedding_effected, embedding_target, dim=-1).sum()
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        if writer: 
            writer.add_scalar("loss", loss.item(), n)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}")

        if n % log_audio_every_n == 0:
            # Save audio
            signal_effected.detach().cpu().write_audio_to_tb("effected", writer, n)
            if writer:
                writer.add_audio("effected", signal_effected.samples[0][0], n, sample_rate=signal_effected.sample_rate)

    # Play final signal with optimized effects parameters
    out_sig = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    out_sig.write(save_dir / "final.wav")

    if writer:
        writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
        writer.close()
    
    return out_sig

def text2fx_params(
    sig: AudioSignal, 
    text: str,   
    channel: Channel,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    log_audio_every_n: int = 25, 
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    save_dir: str = None, # figure out a save path automatically
    params_raw: bool = False,
    params_set: torch.Tensor = None
):
    # ah yes, the max morrison trick of hiding global variables as function members
    # prevents loading the model everytime w/o needing to set it first as global variable
    if not hasattr(text2fx, "msclap"):
        msclap  = MSCLAPWrapper()
        setattr(text2fx, "msclap", msclap)
    else: 
        msclap = text2fx.msclap

    # a save dir for our goods
    if save_dir is None:
        save_dir = create_save_dir(text, sig, RUNS_DIR)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # create a writer for saving stuff to tensorboard
    writer_dir = save_dir / "logs"
    writer_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(writer_dir) #SummaryWriter is tensorboard writer

    # params!
    if params_set is None:
        print('params are set to 0')
        params = torch.nn.parameter.Parameter(
                torch.zeros(sig.batch_size, channel.num_params).to(device) 
            )
    else:
        print('applying user defined set of params')
        # params_rand = torch.randn(signal.batch_size, channel.num_params)
        params = torch.nn.parameter.Parameter(params_set.to(device))

    params.requires_grad=True
    optimizer = torch.optim.Adam([params], lr=lr) #the optimizer

    # log what our initial effect sounds like (w/ random parameters applied)
    if params_raw == True:
        print('applying raw params / no sigmoid - should be all zeros')
        init_sig = channel(sig.clone().to(device), params)
    else:
    # elif params_raw == False:
        print('applying sigmoid-ified params')
        init_sig = channel(sig.clone().to(device), torch.sigmoid(params))

    if writer:
        writer.add_audio("input", sig.samples[0][0], 0, sample_rate=sig.sample_rate)
        writer.add_audio("effected", init_sig.samples[0][0], 0, sample_rate=init_sig.sample_rate)

    sig.clone().cpu().write(save_dir / 'input.wav')

    embedding_target = msclap.get_text_embeddings([text]).detach()
    
    if criterion == "directional_loss":
        audio_in_emb = msclap.preprocess_and_embed(sig.to(device)).detach()
        text_anchor_emb = msclap.get_text_embeddings(["a sound"]).detach()

    # Optimize our parameters by matching effected audio against the target audio
    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        
        # Apply effect with out estimated parameters
        signal_effected = channel(sig.to(device), torch.sigmoid(params.to(device)))

        # Get CLAP embedding for effected audio
        embedding_effected = msclap.preprocess_and_embed(signal_effected) #.get_audio_embeddings takes in preprocessed audio

        # loss
        if criterion == "directional_loss":
            loss = clip_directional_loss(embedding_effected, audio_in_emb, embedding_target, text_anchor_emb).sum()
        elif criterion == "standard": #is neg dot product loss aims to minimize the dot prod b/w dissimilar items, no direction intake
            loss = -(embedding_effected @ embedding_target.T).sum()
        elif criterion == "cosine-sim": # cosine_sim loss aims to maximize the cosine similarity between similar items, normalized
            loss = 1 - torch.cosine_similarity(embedding_effected, embedding_target, dim=-1).sum()
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        if writer: 
            writer.add_scalar("loss", loss.item(), n)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}")

        if n % log_audio_every_n == 0:
            # Save audio
            signal_effected.detach().cpu().write_audio_to_tb("effected", writer, n)
            if writer:
                writer.add_audio("effected", signal_effected.samples[0][0], n, sample_rate=signal_effected.sample_rate)

    # Play final signal with optimized effects parameters
    out_sig = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    out_sig.write(save_dir / "final.wav")

    if writer:
        writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
        writer.close()
    
    return out_sig

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument("--input_audio", type=int, default=5, help="index of example audio file")
    parser.add_argument("--input_audio", type=str, help="path to input audio file")
    parser.add_argument("--text", type=str, help="text prompt for the effect")
    parser.add_argument("--criterion", type=str, default="standard", help="criterion to use for optimization")
    parser.add_argument("--n_iters", type=int, default=600, help="number of iterations to optimize for")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for optimization")

    args = parser.parse_args()

    channel = get_default_channel()
    example_files = load_audio_examples()
    # signal = AudioSignal(example_files[2])
    # signal = AudioSignal(example_files[args.input_audio])
    signal = AudioSignal(args.input_audio)

    text2fx(
        signal, args.text, channel,
        criterion=args.criterion, 
    )
    # text2fx_params(
    #     signal, args.text, channel,
    #     criterion=args.criterion, 
    #     n_iters=args.n_iters, 
    #     lr=args.lr, 

    #     params_raw = False,
    #     # params_set = torch.randn(signal.batch_size, channel.num_params)
    # )