"""
TODO: this should probably use pytest. 
"""
from pathlib import Path
import random

import text2fx
from text2fx import SAMPLE_RATE, Channel, load_audio_examples, get_default_channel, slugify
from audiotools import AudioSignal
import dasp_pytorch
import datetime


#AC remix: trying to log on tensorboard
def test_text2fx_LOG():
    channel = get_default_channel()
    example_files = load_audio_examples()
    signal = AudioSignal(example_files[0])


    runs_dir = Path("runs")

    today = datetime.datetime.now().strftime("%Y-%m-%d")  
    today_dir = runs_dir / today
    today_dir.parent.mkdir(parents=True, exist_ok=True)

    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in ["this sounds like a telephone"]:
            # Apply text2fx
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                save_dir=today_dir / f"criterion-{criterion}" / text_target.replace("this sounds like ", "").replace(" ", "_")
            )

def test_criterion_LOG():
    channel = get_default_channel()
    example_files = load_audio_examples()
    signal = AudioSignal(example_files[0])

    runs_dir = Path("runs")

    today = datetime.datetime.now().strftime("%Y-%m-%d")  
    today_dir = runs_dir / today
    today_dir.parent.mkdir(parents=True, exist_ok=True)

    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in ["this sounds like a telephone"]:
            # Apply text2fx
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                save_dir=today_dir / f"criterion-{criterion}" / text_target.replace("this sounds like ", "").replace(" ", "_")
            )

def test_iterations_LOG():
    """
    text_target = this sounds like a _telephone_
    i_variable: criterion
    n_iters = 1000 (see progression on tensorboard)
    audio_type = voice // example_file[5] - 'please call stella'
    """
    channel = get_default_channel()
    example_files = load_audio_examples()
    signal = AudioSignal(example_files[5])

    runs_dir = Path("runs")
    today = datetime.datetime.now().strftime("%Y-%m-%d")  
    today_dir = runs_dir / today
    today_dir.parent.mkdir(parents=True, exist_ok=True)

    for criterion in ("directional_loss", "cosine-sim", "standard"):
        for text_target in ["this sounds like a telephone"]:
            #step 1: checking dir paths for tensorboard logging
            #logging by DATE > TEXT_TARGET > CRITERION_NAME
            save_dir = today_dir / text_target.replace("this sounds like ", "").replace(" ", "_") / f"criterion-{criterion}"
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            existing_runs = [d for d in save_dir.parent.iterdir() if d.is_dir() and save_dir.name in d.name]
            if len(existing_runs) == 0:
                run_dir = save_dir
            else:
                # If existing runs, find the latest run and increment the number
                latest_run = sorted(existing_runs)[-1]  # Get the latest run directory
                run_num = int(latest_run.stem.split('-')[-1]) + 1
                run_dir = save_dir.parent / f"{save_dir.name}-{run_num}"
            run_dir.mkdir(exist_ok=True, parents=True)

            #step 2: Apply text2fx + log
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                n_iters=1000,
                save_dir=run_dir
            )

def test_text2fx_local():
    channel = get_default_channel()
    # channel = Channel(
    #     # Apply random EQ, Compression, and Gain to a signal
    #     dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
    #     dasp_pytorch.Compressor(sample_rate=SAMPLE_RATE),
    #     dasp_pytorch.Gain(sample_rate=SAMPLE_RATE),
    #     dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE),
        
    #     # Apply random Reverb and Distortion to a signal
    #     # Distortion(sample_rate=SAMPLE_RATE),
    # )
    example_files = load_audio_examples()
    # Initialize our starting parameters
    signal = AudioSignal(example_files[2])

    save_dir = Path("experiments") / "test"

    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in [
            "this sounds like a telephone",
        ]:
            # Apply text2fx
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                save_dir=save_dir / f"criterion-{criterion}" / text_target.replace("this sounds like ", "").replace(" ", "_")
            )


def random_signal(duration):
    example_files = load_audio_examples()
    signal = AudioSignal.salient_excerpt(random.choice(example_files), duration=duration, sample_rate = SAMPLE_RATE)
    return signal

def test_text2fx_batch():
    channel = get_default_channel()
    sig_batch = AudioSignal.batch(
        [random_signal(2).resample(SAMPLE_RATE).to_mono() for x in range(5)]
    )

    save_dir = Path("experiments") / "test"

    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in [
            "this sounds like a telephone",
        ]:
            # Apply text2fx
            signal_effected = text2fx.text2fx(
                sig_batch, text_target, channel,
                criterion=criterion, 
                save_dir=save_dir / f"criterion-{criterion}" / text_target.replace("this sounds like ", "").replace(" ", "_")
            )

if __name__ == "__main__":
    test_iterations_LOG()
    # test_text2fx_batch()