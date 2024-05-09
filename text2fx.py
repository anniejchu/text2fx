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

import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent
ASSETS_DIR = PROJECT_DIR / "assets"
PRETRAINED_DIR = PROJECT_DIR / "pretrained"
DATA_DIR = PROJECT_DIR / "data"
RUNS_DIR = PROJECT_DIR / "runs"

# setting sample rate
SAMPLE_RATE = 44_100  # Resample all audio to a fixed rate, and pass to any effects that need it

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

    @property #AC what does this do? -- hacky thing decorator/annotator, concrete attribute, this is a getter
    def num_params(self):
        return sum([m.num_params for m in self.modules])

    #if you call the object, it automatically calls **forward()** (uses __call__)
    def forward(self, signal: AudioSignal, params: torch.Tensor):

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

def load_audio_examples():
    # Load audio examples
    exts = ["mp3", "wav", "flac"]
    example_files = [list(ASSETS_DIR.rglob(f"*.{e}")) for e in exts]
    example_files = sum(example_files, [])  # Trick to flatten list of lists
    return example_files

""" utility wrapper around MS CLAP model! """
class MSCLAPWrapper:

    def __init__(self):
        self.clap_model = CLAP(version = '2023', use_cuda=True)

    #testing just the clap_model.load_audio() !!
    def resample(self, signal: AudioSignal, resample=True):
        """
        trying to see if resampling step in read_audio (step 1) is the issue, but it seems its from the very beginning w/ reading in the file
        """
        audio_time_series = signal.samples
        resample_rate = self.clap_model.args.sampling_rate
        sample_rate = signal.sample_rate #check

        if resample and resample_rate != sample_rate:
            resampler = T.Resample(sample_rate, resample_rate)
            resampler.to(signal.device)
            audio_time_series = resampler(audio_time_series)
            signal.samples = audio_time_series
            signal.sample_rate = resample_rate
            
        return signal

    def audio_trim(self, audio_time_series, audio_duration, sample_rate):
        audio_time_series = audio_time_series.squeeze(0).squeeze(0)
        if audio_duration*sample_rate >= audio_time_series.shape[0]: #if audio duration is shorter than 7 seconds, repeat samples
            repeat_factor = int(np.ceil((audio_duration*sample_rate) /
                                        audio_time_series.shape[0]))
            # Repeat audio_time_series by repeat_factor to match audio_duration
            audio_time_series = audio_time_series.repeat(repeat_factor)
            # remove excess part of audio_time_series
            audio_time_series = audio_time_series[0:audio_duration*sample_rate]
        else:
            # audio_time_series is longer than predefined audio duration (7s),
            # so audio_time_series is trimmed
            start_index = random.randrange(
                audio_time_series.shape[0] - audio_duration*sample_rate)
            audio_time_series = audio_time_series[start_index:start_index +
                                                audio_duration*sample_rate]
        return audio_time_series.unsqueeze(0).unsqueeze(0).float()

    def preprocess(self, signal: AudioSignal):
        signal_resamp = self.resample(signal) #uses CLAP function on AudioSignal.samples
        signal_resamp.to_mono()
        signal_resamp.samples = self.audio_trim(
            signal_resamp.samples, 
            self.clap_model.args.duration, 
            self.clap_model.args.sampling_rate
        ) #bottleneck - only single sample

        return signal_resamp

    def embed(self, preprocessed_audio: AudioSignal): 
        preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
        return self.clap_model.clap.audio_encoder(preprocessed_audio)[0]

    def preprocess_and_embed(self, signal: AudioSignal):
        return self.embed(self.preprocess(signal).samples)

    def get_text_embeddings(self, texts):
        return self.clap_model.get_text_embeddings(texts)

# just for saving our text prompts as filename safely!
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

def create_save_dir(text, sig, runs_dir):
    """ 
    create a save folder for our current run.  
    """
    # lets make a runs dir
    # make a subfolder under runs with today's date in YYYY-MM-DAY format
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    # run name should be text prompt + a number if there are multiple runs with the same prompt
    run_name = f"{slugify(text)}"    
    today_dir = runs_dir / today
    today_dir.parent.mkdir(parents=True, exist_ok=True)

    # see if we have any dirs in there with the same name
    existing_runs = [d for d in today_dir.parent.iterdir() if d.is_dir() and run_name in d.stem]

    if len(existing_runs) == 0:
        save_dir = today_dir / f"{run_name}-001"
    else:
        latest_run = sorted(existing_runs, key=lambda x: x.split('-')[-1])[-1]
        run_num = int(latest_run.split('-')[-1]) + 1
        save_dir.stem = today_dir / f"{run_name}-{run_num}"

    save_dir.mkdir(exist_ok=True, parents=True)
    return save_dir

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
        # Apply random EQ, Compression, and Gain to a signal
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
    save_dir: str = "auto" # figure out a save path automatically
):
    # ah yes, the max morrison trick of hiding global variables as function members
    if not hasattr(text2fx, "msclap"):
        msclap  = MSCLAPWrapper()
        setattr(text2fx, "msclap", msclap)
    else: 
        msclap = text2fx.msclap

    # a save dir for our goods
    if save_dir == "auto":
        save_dir = create_save_dir(text, sig, RUNS_DIR)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

    # create a writer for saving stuff to tensorboard
    writer_dir = save_dir / "logs"
    writer_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(writer_dir) 

    # params!
    params = torch.nn.parameter.Parameter(
        torch.zeros(sig.batch_size, channel.num_params).to(device)
    )
    params.requires_grad=True
    # the optimizer!
    optimizer = torch.optim.Adam([params], lr=lr)

    # log what our initial effect sounds like
    init_sig = channel(sig.clone().to(device), torch.sigmoid(params))
    if writer:
        # sig.detach().cpu().write_audio_to_tb("input", writer, 0)
        # init_sig.detach().cpu().write_audio_to_tb("effected", writer, 0)
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
        # signal_effected = channel(signal, torch.sigmoid(params))
        signal_effected = channel(sig.to(device), torch.sigmoid(params.to(device)))

        # Get CLAP embedding for effected audio
        embedding_effected = msclap.preprocess_and_embed(signal_effected) #.get_audio_embeddings takes in preprocessed audio

        # Compute distance between our effected audio embedding vector and the target text 
        # embedding vector. We use cosine distance (essentially the negative dot product
        # between vectors). When two vectors are "aligned" (pointing in the same direction), 
        # their dot product will be high. We thus penalize a small dot product to try to
        # bring our audio vector "into alignment" with the text vector.
        if criterion == "directional_loss":
            loss = clip_directional_loss(embedding_effected, audio_in_emb, embedding_target, text_anchor_emb)
        elif criterion == "standard":
            loss = -(embedding_effected @ embedding_target.T).sum()
        elif criterion == "cosine-sim":
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
                writer.add_audio("effected", signal_effected.samples, n, sample_rate=signal_effected.sample_rate)

    # Play final signal with optimized effects parameters
    out_sig = channel(sig.clone().to(device), torch.sigmoid(params)).clone().detach().cpu()
    out_sig.write(save_dir / "final.wav")

    if writer:
        # out_sig.write_audio_to_tb("final", writer, n_iters)
        writer.add_audio("final", out_sig.samples[0][0], n_iters, sample_rate=out_sig.sample_rate)
        writer.close()
    
    return out_sig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_audio", type=str, help="path to input audio file")
    parser.add_argument("--text", type=str, help="text prompt for the effect")
    parser.add_argument("--output_folder", type=str, help="folder to save output audio to")
    parser.add_argument("--criterion", type=str, default="standard", help="criterion to use for optimization")

    args = parser.parse_args()

    channel = get_default_channel()
    signal = AudioSignal(args.input_audio)

    text2fx(
        signal, args.text, channel,
        criterion=args.criterion, 
        save_dir=args.output_folder
    )
