"""
TODO: this should probably use pytest. 
"""
from pathlib import Path
import random

import text2fx
from text2fx import SAMPLE_RATE, Channel, load_audio_examples, get_default_channel
from audiotools import AudioSignal
import dasp_pytorch

#AC remix: trying to log on tensorboard
from text2fx import SAMPLE_RATE, Channel, load_audio_examples, get_default_channel, create_save_dir #MSCLAPWrapper
from torch.utils.tensorboard import SummaryWriter

from text2fx import text2fx
from torch.utils.tensorboard import SummaryWriter

def test_text2fx_LOG():
    channel = get_default_channel()
    example_files = load_audio_examples()
    signal = AudioSignal(example_files[0])

    # save_dir = Path("experiments") / "test"

    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in ["this sounds like a telephone"]:
            # Apply text2fx
            text2fx(
                signal, text_target, channel,
                criterion=criterion,
                # writer_dir=tensorboard_writer.log_dir,  # Pass the log directory to text2fx
                # save_dir=save_dir / f"criterion-{criterion}" / text_target.replace("this sounds like ", "").replace(" ", "_")
            )



def test_text2fx():
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
    test_text2fx_LOG()
    # test_text2fx_batch()