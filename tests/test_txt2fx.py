"""
TODO: this should probably use pytest. 
"""
from pathlib import Path

import text2fx
from text2fx import SAMPLE_RATE, Channel, load_audio_examples
from audiotools import AudioSignal
import dasp_pytorch


def test_text2fx():
    channel = Channel(
        # Apply random EQ, Compression, and Gain to a signal
        dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
        dasp_pytorch.Compressor(sample_rate=SAMPLE_RATE),
        dasp_pytorch.Gain(sample_rate=SAMPLE_RATE),
        dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE),
        
        # Apply random Reverb and Distortion to a signal
        # Distortion(sample_rate=SAMPLE_RATE),
    )
    example_files = load_audio_examples()
    # Initialize our starting parameters
    signal = AudioSignal(example_files[0])

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


if __name__ == "__main__":
    test_text2fx()