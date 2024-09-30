from pathlib import Path
from tqdm import tqdm

import torch
import numpy as np
import audiotools as at
import dasp_pytorch
from audiotools import AudioSignal
import json
from typing import Union, List

# from torch.utils.tensorboard import SummaryWriter

# from msclap import CLAP

# from text2fx.core import Channel, AbstractCLAPWrapper, Distortion, create_save_dir, preprocess_audio
# from text2fx.constants import RUNS_DIR, SAMPLE_RATE, DEVICE



from torch.utils.tensorboard import SummaryWriter

from text2fx.core import Channel, AbstractCLAPWrapper, Distortion, create_save_dir, preprocess_audio, download_file
from text2fx.constants import RUNS_DIR, SAMPLE_RATE, DEVICE, PRETRAINED_DIR


import torchaudio.transforms as T
import numpy as np
import dasp_pytorch
from audiotools import AudioSignal
from typing import Iterable
import random

# from msclap import CLAP
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
device = DEVICE #torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

def detensor_dict(input_dict: dict) -> dict:
    output_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else
        {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in value.items()} if isinstance(value, dict) else value for key, value in input_dict.items()}
    return output_dict

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
    )

def text2fx_paper(
    model_name: str,
    sig: AudioSignal, 
    text: Union[str, List[str]],   
    channel: Channel,
    device: str = "cuda" if torch.cuda.is_available() else "cpu", 
    log_audio_every_n: int = 25, 
    lr: float = 1e-2, 
    n_iters: int = 650,
    criterion: str = "standard", 
    save_dir: str = None, # figure out a save path automatically,
    params_init_type: str = "random",
    # seed_i: int = 0,
    roll_amt: int = None,
    export_audio: bool = False,
    log_tensorboard: bool = False,
):
    clap = get_model(model_name)
    if log_tensorboard or export_audio:
        if not save_dir:
            save_dir = create_save_dir(f'{text}_paper', RUNS_DIR)
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

    # create a writer for saving stuff to tensorboard
    if log_tensorboard:
        writer_dir = save_dir / "logs"
        writer_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(writer_dir) #SummaryWriter is tensorboard writer
    else:
        writer = False
    # params!
    if params_init_type=='zeros':
        params = torch.nn.parameter.Parameter(
            torch.zeros(sig.batch_size, channel.num_params).to(device) 
        )
    elif params_init_type=='random':
        params = torch.nn.parameter.Parameter(
            #params_single.repeat(sig.batch_size, 1).to(device)
            torch.randn(sig.batch_size, channel.num_params).to(device) 
            # torch.normal(mean=0, std=0.5, size=(sig.batch_size, channel.num_params)).to(device)
        )
    elif params_init_type=='super_random':
        params = torch.nn.parameter.Parameter(
            (torch.randn(sig.batch_size, channel.num_params).to(device) * 8) 
        )
        print('sueprandom')

    else:
        raise ValueError
    
    # Log the model, torch amount, starting parameters, and their values
    if log_tensorboard or export_audio:
        log_file = save_dir / f"experiment_log.txt"
        with open(log_file, "w") as log:
            log.write(f"Model: {model_name}\n")
            log.write(f"Channel: {channel.modules}\n")
            log.write(f"Learning Rate: {lr}\n")
            log.write(f"Number of Iterations: {n_iters}\n")
            log.write(f"Criterion: {criterion}\n")
            log.write(f"Params Initialization Type: {params_init_type}\n")
            log.write(f"Starting Params Values: {params.data.cpu().numpy()}\n")
            log.write(f"Starting Params Values (post sigmoid): {torch.sigmoid(params).data.cpu().numpy()}\n")

            log.write(f"Custom roll?: {roll_amt}\n")
            log.write("="*40 + "\n")

    # in_params = params.data.cpu()
    # in_params_dict = channel.save_params_to_dict(in_params)

    params.requires_grad=True
    # the optimizer!
    optimizer = torch.optim.Adam([params], lr=lr)

    #preprocessing initial sample
    sig = preprocess_audio(sig).to(device)
    # log what our initial effect sounds like (w/ random parameters applied)

    init_sig = channel(sig.clone().to(device), torch.sigmoid(params))
    if writer:
        writer.add_audio("input", sig.samples[0][0], 0, sample_rate=sig.sample_rate)
        writer.add_audio("effected", init_sig.samples[0][0], 0, sample_rate=init_sig.sample_rate)
    # sig.clone().cpu().write(save_dir / 'input.wav')
    if export_audio: #starting audio
        if sig.batch_size == 1:
            init_sig_path = Path(init_sig.path_to_file)
            sig.clone().detach().cpu().write(save_dir / f'{init_sig_path.stem}_input.wav')
            init_sig.detach().cpu().write(save_dir / f'{init_sig_path.stem}_starting.wav')

        else:
            for i, s in enumerate(init_sig):
                sig[i].clone().detach().cpu().write(save_dir / f'{init_sig.path_to_file[i].stem}_input.wav')
                init_sig[i].detach().cpu().write(save_dir / f'{init_sig.path_to_file[i].stem}_starting.wav')

    if isinstance(text, str):
        text = [text]
    assert len(text) == sig.batch_size or len(text) == 1

    if len(text) < sig.batch_size:
        text = text * sig.batch_size

    # Preprocess text
    text_processed = [
        f"this sound is {t}" for t in text
    ]
    embedding_target = clap.get_text_embeddings(text_processed).detach()
    # print(embedding_target)

    if criterion == "directional_loss":
        audio_in_emb = clap.get_audio_embeddings(sig.to(device)).detach()

        text_neg_processed = [
            f"this sound is not {t}" for t in text
        ]
        text_anchor_emb = clap.get_text_embeddings(text_neg_processed).detach()

    final_losses = []
    json_log_path = save_dir / "params_log.json"  # Path to save the JSON file

    # Optimize our parameters by matching effected audio against the target audio
    pbar = tqdm(range(n_iters), total=n_iters)
    for n in pbar:
        
        # Apply effect with out estimated parameters
        sig_roll = sig.clone()

        if roll_amt or roll_amt == 0:
            roll_amount = torch.randint(-roll_amt, roll_amt + 1, (sig_roll.batch_size,))
        else:
            roll_amount = torch.randint(0, sig_roll.signal_length, (sig_roll.batch_size,))

        if log_tensorboard or export_audio:
            with open(log_file, "a") as log:
                log.write(f"Iteration {n}: roll_amount: {roll_amount.cpu().numpy()}\n")

        for i in range(sig_roll.batch_size):
            rolled = torch.roll(sig_roll.samples[i], shifts=roll_amount[i].item(), dims=-1)
            # print(rolled)
            sig_roll.samples[i:i+1] = rolled

        signal_effected = channel(sig_roll.to(device), torch.sigmoid(params.to(device)))
        signal_effected_original = channel(sig.clone().to(device), torch.sigmoid(params.to(device)))

        # Get CLAP embedding for effected audio
        embedding_effected = clap.get_audio_embeddings(signal_effected) #.get_audio_embeddings takes in preprocessed audio

        # loss
        if criterion == "directional_loss":
            batch_loss = clip_directional_loss(embedding_effected, audio_in_emb, embedding_target, text_anchor_emb)
            # loss = clip_directional_loss(embedding_effected, audio_in_emb, embedding_target, text_anchor_emb).mean()
        elif criterion == "standard": #is neg dot product loss aims to minimize the dot prod b/w dissimilar items, no direction intake
            batch_loss = -(embedding_effected @ embedding_target.T)
            # loss = -(embedding_effected @ embedding_target.T).mean()
        elif criterion == "cosine-sim": # cosine_sim loss aims to maximize the cosine similarity between similar items, normalized
            batch_loss = 1 - torch.cosine_similarity(embedding_effected, embedding_target, dim=-1)
            # loss = 1 - torch.cosine_similarity(embedding_effected, embedding_target, dim=-1).mean()
        else:
            raise ValueError(f"Criterion {criterion} not recognized")
        
        loss = batch_loss.mean()
        if writer: 
            writer.add_scalar("loss", loss.item(), n)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"step: {n+1}/{n_iters}, loss: {loss.item():.3f}")

        #saving last batch_loss
        if n == n_iters - 1:
            final_losses = batch_loss.detach().cpu().numpy()

        if log_tensorboard:
            if n % log_audio_every_n == 0:
                # Save audio
                signal_effected.detach().cpu().ensure_max_of_audio().write_audio_to_tb("effected", writer, n)
                if writer:
                    writer.add_audio("effected", signal_effected.clone().ensure_max_of_audio().samples[0][0], n, sample_rate=signal_effected.sample_rate)
        if n % 100 == 0:
            params_i = params.detach().cpu()
            out_params_dict = channel.save_params_to_dict(params.detach().cpu())
            print(out_params_dict)
            with open(json_log_path, "a") as json_log_file:
                json.dump({"iteration": n, "params": detensor_dict(out_params_dict)}, json_log_file)
                json_log_file.write("\n")  # For better readability in the file
                json.dump({"iteration": n, "raw_params": params_i.tolist()}, json_log_file)
                json_log_file.write("\n")  # For better readability in the file

            
            signal_effected_original.detach().cpu().normalize(-24).write(save_dir / f'{init_sig_path.stem}_{n}.wav')
            # out_sig.detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')

    if log_tensorboard or export_audio:
        with open(log_file, "a") as log:
            log.write(f"ENDING Params Values: {params.data.cpu().numpy()}\n")
    
    min_loss_index = int(np.argmin(final_losses))
    # breakpoint()
    # Play final signal with optimized effects parameters
    out_sig = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    # out_sig = preprocess_audio(out_sig)
    
    final_params = params.detach().cpu()
    final_params_dict = channel.save_params_to_dict(final_params)
    with open(json_log_path, "a") as json_log_file:
        json.dump({"iteration": 'end', "params": detensor_dict(final_params_dict)}, json_log_file)
        json_log_file.write("\n")  # For better readability in the file
    if export_audio:
        if sig.batch_size == 1:
            out_sig.detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')
            # out_sig.clone().detach().cpu().write(save_dir / f'{init_sig_path.stem}_final.wav')
        else:
            for i, s in enumerate(out_sig):
                i_init_sig_path = Path(init_sig.path_to_file[i])
                out_sig[i].detach().cpu().write(save_dir / f'{i_init_sig_path.stem}_final.wav')

    # out_sig.write(save_dir / "final.wav")
    if writer:
        writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
        writer.close()
    
    init_sig_out = init_sig.detach().cpu()
    # init_sig_out = preprocess_audio(init_sig_out)

    return out_sig, final_params, final_params_dict#, final_losses, min_loss_index, init_sig_out#params.detach().cpu()
    # return out_sig, final_params, final_params_dict, final_losses, min_loss_index, init_sig_out#params.detach().cpu()

# def test():
#     sig = AudioSignal("assets/speech_examples/VCTK_p225_001_mic1.flac").resample(44_100).to_mono().normalize(-24)
#     # sig = AudioSignal("assets/audealize_examples/guitar.wav", duration=3).resample(44_100).to_mono().normalize(-24)

#     # channel = Channel(dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE))
#     channel = Channel(dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE))

#     text2fx_paper(
#         model_name='ms_clap',
#         sig = sig,
#         text = 'hollow and far-away',
#         channel = channel,
#         n_iters=650,
#         criterion='directional_loss',#'cosine-sim',
#         export_audio=True,
#         log_tensorboard=True,
#         params_init_type='zeros'
#     )

# test()
#     text2fx_paper(
#     model_name: str,
#     sig: AudioSignal, 
#     text: Union[str, List[str]],   
#     channel: Channel,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu", 
#     log_audio_every_n: int = 25, 
#     lr: float = 1e-2, 
#     n_iters: int = 650,
#     criterion: str = "standard", 
#     save_dir: str = None, # figure out a save path automatically,
#     params_init_type: str = "random",
#     # seed_i: int = 0,
#     roll_amt: int = None,
#     export_audio: bool = False,
#     log_tensorboard: bool = False,
# ):

# if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()

    # # parser.add_argument("--input_audio", type=int, default=5, help="index of example audio file")
    # parser.add_argument("--model_name", type=str, help="choose either 'laion_clap' or 'ms_clap'")
    # parser.add_argument("--input_audio", type=str, help="path to input audio file")
    # parser.add_argument("--text", type=str, help="text prompt for the effect")
    # parser.add_argument("--criterion", type=str, default="cosine-sim", help="criterion to use for optimization")
    # parser.add_argument("--n_iters", type=int, default=600, help="number of iterations to optimize for")
    # parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for optimization")
    # parser.add_argument("--save_dir", type=str, default=None, help="path to export audio file")
    # parser.add_argument("--params_init_type", type=str, default='zeros', help="enter params init type")
    # parser.add_argument("--roll_amt", type=int, default=None, help="range of # of samples for rolling action")
    # parser.add_argument("--export_audio", type=bool, default=False, help="export audio?")
    # parser.add_argument("--log_tensorboard", type=bool, default=False, help="log tensorboard?")


    # # args = parser.parse_args()

    # channel = Channel(dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE))

    # text2fx(
    #     model_name=args.model_name, 
    #     sig=AudioSignal(args.input_audio), 
    #     text=args.text, 
    #     channel=args.channel,
    #     criterion=args.criterion, 
    #     save_dir=args.save_dir,
    #     params_init_type=args.params_init_type,
    #     roll_amt=args.roll_amt,
    #     export_audio=args.export_audio,
    #     log_tensorboard=args.log_tensorboard
    # )
