from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import audiotools as at
import dasp_pytorch
from audiotools import AudioSignal
from typing import Iterable
import random
from torch.utils.tensorboard import SummaryWriter

# from msclap import CLAP

from text2fx.core import Channel, AbstractCLAPWrapper, Distortion, load_audio_examples, DEVICE, create_save_dir, RUNS_DIR

import matplotlib.pyplot as plt

"""
EX CLI USAGE
python -m text2fx --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
                 --text "this sound is happy" \
                 --criterion "cosine-sim" \
                 --n_iters 600 \
                 --lr 0.01 
                 --params_init_type "zeros"
                 --
"""
SAMPLE_RATE = 44100
device = DEVICE #torch.device("cuda:0") if torch.cuda.is_available() else "cpu"


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
        dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
        dasp_pytorch.Compressor(sample_rate=SAMPLE_RATE),
        dasp_pytorch.Gain(sample_rate=SAMPLE_RATE),
        dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE),
        
        # Distortion(sample_rate=SAMPLE_RATE),
    )

def text2fx(
    model_name: str,
    sig: AudioSignal, 
    text: str,   
    channel: Channel,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    log_audio_every_n: int = 25, 
    lr: float = 1e-2, 
    n_iters: int = 600,
    criterion: str = "standard", 
    save_dir: str = None, # figure out a save path automatically,
    params_init_type: str = "zeros",
    seed_i: int = 0,
    roll: str = 'all',
    roll_amt: int = 1000,
):
    # ah yes, the max morrison trick of hiding global variables as function members
    # prevents loading the model everytime w/o needing to set it first as global variable
    # if model=='ms_clap':
    at.util.seed(seed_i)
    clap = get_model(model_name)

    # a save dir for our goods
    if save_dir is None:
        save_dir = create_save_dir(text, sig, RUNS_DIR)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    print(save_dir)

    # create a writer for saving stuff to tensorboard
    writer_dir = save_dir / "logs"
    writer_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(writer_dir) #SummaryWriter is tensorboard writer

    # params!
    # NOTE: these aren't actually initialized to "zeros" since the we'll apply a sigmoid which will shift this up right? 
    if params_init_type=='zeros':
        params = torch.nn.parameter.Parameter(
            torch.zeros(sig.batch_size, channel.num_params).to(device) 
        )
    elif params_init_type=='random':
        params = torch.nn.parameter.Parameter(
            torch.randn(sig.batch_size, channel.num_params).to(device) 
        )
    else:
        raise ValueError
    
    # Log the model, torch amount, starting parameters, and their values
    log_file = save_dir / f"experiment_log.txt"
    with open(log_file, "a") as log:
        log.write(f"Model: {model_name}\n")
        log.write(f"Learning Rate: {lr}\n")
        log.write(f"Number of Iterations: {n_iters}\n")
        log.write(f"Criterion: {criterion}\n")
        log.write(f"Params Initialization Type: {params_init_type}\n")
        log.write(f"Seed: {seed_i}\n")
        log.write(f"Starting Params Values: {params.data.cpu().numpy()}\n")
        log.write(f"Starting Params Values (post sigmoid): {torch.sigmoid(params).data.cpu().numpy()}\n")

        log.write(f"roll?: {roll}, if custom: range is +/-{roll_amt}\n")
        log.write("="*40 + "\n")

    params.requires_grad=True
    # the optimizer!
    optimizer = torch.optim.Adam([params], lr=lr)

    #preprocessing initial sample
    sig = sig.resample(44100).normalize(-24) 

    # log what our initial effect sounds like (w/ random parameters applied)
    init_sig = channel(sig.clone().to(device), torch.sigmoid(params))
    if writer:
        writer.add_audio("input", sig.samples[0][0], 0, sample_rate=sig.sample_rate)
        writer.add_audio("effected", init_sig.samples[0][0], 0, sample_rate=init_sig.sample_rate)

    sig.clone().cpu().write(save_dir / 'input.wav')
    init_sig.clone().detach().cpu().write(save_dir / 'starting.wav')

    embedding_target = clap.get_text_embeddings([text]).detach()
    
    if criterion == "directional_loss":
        audio_in_emb = clap.get_audio_embeddings(sig.to(device)).detach()
        text_anchor_emb = clap.get_text_embeddings([f"this sound is not {text}"]).detach()

    # Optimize our parameters by matching effected audio against the target audio
    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        
        # Apply effect with out estimated parameters
        sig_roll = sig.clone()

        if roll == 'none':
            roll_amount = torch.zeros(sig_roll.batch_size, dtype=torch.int64)
        elif roll == 'custom':
            roll_amount = torch.randint(-roll_amt, roll_amt, (sig_roll.batch_size,))
        elif roll == 'all':
            roll_amount = torch.randint(0, sig_roll.signal_length, (sig_roll.batch_size,))
        else:
            raise ValueError('choose roll amount')

        with open(log_file, "a") as log:
            log.write(f"Iteration {n}: roll_amount: {roll_amount.cpu().numpy()}\n")

        for i in range(sig_roll.batch_size):
            # breakpoint()
            rolled = torch.roll(sig_roll.samples[i], shifts=roll_amount[i].item(), dims=-1)
            # print(rolled)
            sig_roll.samples[i:i+1] = rolled

        signal_effected = channel(sig_roll.to(device), torch.sigmoid(params.to(device)))

        # Get CLAP embedding for effected audio
        embedding_effected = clap.get_audio_embeddings(signal_effected) #.get_audio_embeddings takes in preprocessed audio

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
            signal_effected.detach().cpu().ensure_max_of_audio().write_audio_to_tb("effected", writer, n)
            if writer:
                writer.add_audio("effected", signal_effected.clone().ensure_max_of_audio().samples[0][0], n, sample_rate=signal_effected.sample_rate)

    with open(log_file, "a") as log:
        log.write(f"ENDING Params Values: {params.data.cpu().numpy()}\n")
        
    # Play final signal with optimized effects parameters
    out_sig = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    out_sig.write(save_dir / "final.wav")

    # also exporting normalized output
    # out_sig1 = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu().normalize(-24)
    # out_sig1.write(save_dir / "final_normalized.wav")

    if writer:
        writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
        writer.close()


    
    return out_sig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # parser.add_argument("--input_audio", type=int, default=5, help="index of example audio file")
    parser.add_argument("--model_name", type=str, help="choose either 'laion_clap' or 'ms_clap'")
    parser.add_argument("--input_audio", type=str, help="path to input audio file")
    parser.add_argument("--text", type=str, help="text prompt for the effect")
    parser.add_argument("--criterion", type=str, default="cosine-sim", help="criterion to use for optimization")
    parser.add_argument("--n_iters", type=int, default=600, help="number of iterations to optimize for")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for optimization")
    parser.add_argument("--save_dir", type=str, default=None, help="path to export audio file")
    parser.add_argument("--params_init_type", type=str, default='zeros', help="enter params init type")
    parser.add_argument("--seed_i", type=int, default=1, help="enter a number seed start")
    parser.add_argument("--roll", type=str, default='all', help="to roll or not to roll")
    parser.add_argument("--roll_amt", type=int, default=1000, help="range of # of samples for rolling action")


    args = parser.parse_args()

    channel = Channel(dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE))

    text2fx(
        model_name=args.model_name, 
        sig=AudioSignal(args.input_audio), 
        text=args.text, 
        channel=channel,
        criterion=args.criterion, 
        save_dir=args.save_dir,
        params_init_type=args.params_init_type,
        seed_i=args.seed_i,
        roll=args.roll,
        roll_amt=args.roll_amt

    )
