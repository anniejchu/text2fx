"""
TODO: this should probably use pytest. 
"""
from pathlib import Path
import random

import text2fx
from text2fx import SAMPLE_RATE, Channel, load_audio_examples, get_default_channel, slugify, Distortion
from audiotools import AudioSignal
import dasp_pytorch
import datetime
import re
import torch

def test_TextTargets(textTargets):
    """
    testing locally different textTargets and criterion
    default channel
    """
    prefix = 'this is the sound of '
    y = [prefix + x for x in textTargets]
    # print(y)

    channel = get_default_channel()

    example_files = load_audio_examples()
    signal = AudioSignal(example_files[2])

    save_dir = Path("experiments") / "test512"

    for criterion in ("directional_loss", "cosine-sim"):
        for text_target in y:
            # Apply text2fx
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                save_dir = save_dir / text_target.replace(prefix, "").replace(" ", "_") / f"criterion-{criterion}"
            )

def test_FX_text_congruence(textTargets, sig_type='voice'):
    """
    Distortion effect, set parameter (corresponding to gain db) to 0 to start. 
    Then prompt with textTargets = "distorted", "clean", "crisp", "degraded", etc.

    Equalizer effect, set parameters corresponding to "gain" controls all to the same value (e.g. 0) to start. 
    Then prompt with "sharp", "boomy", "warm", "tinny", etc.

    Pulling from: https://interactiveaudiolab.github.io/assets/papers/zheng_seetharaman_pardo_acmmm.pdf
    """
    prefix = 'this is the sound of '
    y = [prefix + x for x in textTargets]

    #testing just EQ
    fx_name = 'COMPRESSION'
    channel = Channel(
        dasp_pytorch.Compressor(sample_rate=SAMPLE_RATE),
    )

    example_files = load_audio_examples()
    if sig_type == 'voice':
        signal = AudioSignal(example_files[5])
    elif sig_type == 'music':
        signal = AudioSignal(example_files[2])
    else:
        raise ValueError(f"sig_type {sig_type} not recognized")

    save_dir = Path("experiments") / "test512_congruence"

    for criterion in ("cosine-sim",):
        for text_target in y:
            run_dir = save_dir / f"{fx_name}" / f"{sig_type}" / text_target.replace(prefix, "").replace(" ", "_") /f"criterion-{criterion}"
            # Apply text2fx
            signal_effected = text2fx.text2fx_params( #using the text2fx_params version
                signal, text_target, channel,
                criterion=criterion, 
                save_dir = run_dir,
                params_raw = True #makes sure that the starting initialization parameters are 0
            )


def test_LOG_BIG(local: bool = True):
    """
    text_target = this sounds like a _telephone_
    i_variable: criterion
    audio_type = voice // example_file[5] - 'please call stella'
    """
    prefix = 'this sounds like '
    textTargets = ['a telephone', 'a church']
    y = [prefix + x for x in textTargets]
    channel = get_default_channel()
    example_files = load_audio_examples()
    signal = AudioSignal(example_files[5])

    for criterion in ("cosine-sim",):
        for text_target in y:
            if not local:
                print('TO TENSORBOARD')
                runs_dir = Path("runs")
                today = datetime.datetime.now().strftime("%Y-%m-%d")  
                today_dir = runs_dir / today
                today_dir.parent.mkdir(parents=True, exist_ok=True)
                #where we'd adapt directory name
                save_dir = today_dir / text_target.replace(prefix, "").replace(" ", "_") / f"criterion-{criterion}"
                save_dir.parent.mkdir(parents=True, exist_ok=True)

                existing_runs = [d for d in save_dir.parent.iterdir() if d.is_dir() and save_dir.name in d.name]
                if len(existing_runs) == 0:
                    run_dir = save_dir
                else:
                    # If existing runs, find the latest run and increment the number
                    latest_run = sorted(existing_runs)[-1]  # Get the latest run directory
                    match = re.search(r'-(\d+)$', latest_run.stem)  # Find the last digits in the name
                    if match:
                        run_num = int(match.group(1)) + 1
                    else:
                        run_num = 1  # No number found, start from 1
                    run_dir = save_dir.parent / f"{save_dir.name}-{run_num}"
                run_dir.mkdir(exist_ok=True, parents=True)
            else:
                print('LOCAL')
                save_dir = Path("experiments") / "test512_localtensorboardfiles"
                run_dir = save_dir / text_target.replace(prefix, "").replace(" ", "_") /f"criterion-{criterion}"

            # #step 2: Apply text2fx + log
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                n_iters=1000,
                save_dir=run_dir
            )

def test_criterion_LOG():
    """
    text_target = this sounds like a [_telephone_, etc, etc]
    i_variable: criterion
    n_iters = 1000 (see progression on tensorboard)
    audio_type = voice // example_file[5] - 'please call stella'
    """
    prefix = 'this sounds like '
    textTargets = ['a telephone', 'a church', 'underwater']
    y = [prefix + x for x in textTargets]
    channel = get_default_channel()
    example_files = load_audio_examples()
    signal = AudioSignal(example_files[5])

    comment_dir = "varied_criterion_textTarget"

    runs_dir = Path("runs")
    today = datetime.datetime.now().strftime("%Y-%m-%d")  
    today_dir = runs_dir / today / comment_dir
    today_dir.parent.mkdir(parents=True, exist_ok=True)

    for criterion in ("directional_loss", "cosine-sim", "standard"):
        for text_target in y:
            #step 1: checking dir paths for tensorboard logging
            #logging by DATE > TEXT_TARGET > CRITERION_NAME
            save_dir = today_dir / text_target.replace(prefix, "").replace(" ", "_") / f"criterion-{criterion}"
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



def test_iters(textTargets, num_experiments, sig_type='voice', title: str= '513_test'):
    """ trying to make a helper function"""
    prefix = 'this is the sound of '
    y = [prefix + x for x in textTargets]

    #testing just EQ
    fx_name = 'EQ+REVERB'
    channel = Channel(
        dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
        dasp_pytorch.NoiseShapedReverb(sample_rate=SAMPLE_RATE),
    )
    example_files = load_audio_examples()
    if sig_type == 'voice':
        signal = AudioSignal(example_files[5])
    elif sig_type == 'music':
        signal = AudioSignal(example_files[2])
    else:
        raise ValueError(f"sig_type {sig_type} not recognized")

    save_dir = Path("experiments") / title

    for criterion in ("cosine-sim",):
        for text_target in y:
            for i in range(num_experiments):
                torch.manual_seed(i)  # Change 'i' to the seed you want to use
                params_set = torch.randn(signal.batch_size, channel.num_params)

                run_dir = save_dir / f"{criterion}" / f"{sig_type}" / text_target.replace(prefix, "").replace(" ", "_") /f"seed-{i}"
                # Apply text2fx
                signal_effected = text2fx.text2fx_params( #using the text2fx_params version
                    signal, text_target, channel,
                    criterion=criterion, 
                    save_dir = run_dir,
                    params_raw = False, #makes sure that the starting initialization parameters are put thru sigmoid
                    params_set = params_set
                )

def test_text2fx_515(textTargets):
    prefix = 'this sound is '
    y = [prefix + x for x in textTargets]
    channel = Channel(
        # Apply random EQ, Compression, and Gain to a signal
        dasp_pytorch.ParametricEQ(sample_rate=SAMPLE_RATE),
    )
    # example_files = load_audio_examples()
    # signal = AudioSignal(example_files[5])

    audio_file = "assets/speech_examples/VCTK_p225_001_mic1.flac"
    signal = AudioSignal(audio_file)
    # Initialize our starting parameters

    save_dir = Path("experiments") / "audealize_comp" / "speech" / "comparatives" 
    for criterion in ("cosine-sim",):
        for text_target in y:
            # Apply text2fx
            signal_effected = text2fx.text2fx(
                signal, text_target, channel,
                criterion=criterion, 
                # save_dir = save_dir / text_target
                save_dir=save_dir / text_target.replace("this sounds like ", "").replace(" ", "_")
            )


if __name__ == "__main__":
    top10_eq = ["warmer", "colder", "softer", "louder", "happier", "brighter", "more soothing", "harsher", "heavier", "cooler"]
    test_text2fx_515(top10_eq)
    # wordlist = ['cold', 'youthful', 'relaxed', 'energetic', 'muddled'] #from audealize
    # test_iters(wordlist, 5, title='vary_init_param_seeds', sig_type='music')

    # testing123(['underwater', 'a telephone', 'a explosion'])
    # for t in ['music', 'voice']:
    #     test_FX_text_congruence(["full", "sharp", "subtle", "fuzzy"], sig_type=t)

    # test_LOG_BIG(local=False)
    # test_criterion_LOG()
    # test_text2fx_batch()