
import audiotools as at
import gradio as gr
import numpy as np

from pathlib import Path
from text2fx.text2fx_app import text2fx_paper as text2fx
import text2fx.core as tc
import text2fx.core_plotting as tcplot
import torch
import os
from process_file_from_params import normalize_param_dict
import uuid
import shutil
import random
import string

"""
TODOS
1. remove hardcoded paths
2. add in create_save_dir
3. improve ui to match more of text2EQ
"""
def generate_random_string():
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15)))
    return f'"{random_str}"'
def generate_random_nums():
    random_str = ''.join(random.choices(string.digits, k=random.randint(5, 15)))
    return f'"{random_str}"'
def extract_label(k):
    parts = k.split('_')  # Split the string by '_'
    label = '_'.join(parts[-2:])  # Join the last two parts
    return label

# TODO: REPLACE WITH ADAPTIVE RANDOM PATH GENERATOR 
ARTIFACTS_DIR = Path('/home/annie/research/text2fx/runs/text2fx_app_artifacts')
shutil.rmtree(ARTIFACTS_DIR)
ARTIFACTS_DIR.mkdir()


def process_fn(data):
    print(data[input_audio])
    print(data[text])
    print(data[criterion])
    print(data[fx_chain])
    channel = tc.create_channel(data[fx_chain])
    # breakpoint()
    in_sig = at.AudioSignal(data[input_audio]).to_mono()
    output_sig, out_params, out_params_dict = text2fx(
        model_name = 'ms_clap',
        sig_in = in_sig, 
        text=data[text],
        export_audio=True,
        channel=channel,
        criterion=data[criterion],
        params_init_type=data[params_init_type],
        n_iters=data[num_iters],
        # save_dir='/tmp/gradio/f6bc142ba5459e2e8db732b3665face1592bdaa2/'
    )
    assert output_sig.path_to_file is not None

    in_path_to_file = os.path.join(os.path.dirname(output_sig.path_to_file), os.path.basename(output_sig.path_to_file).replace('_final', '_input'))

    plot_path = ARTIFACTS_DIR/f'{uuid.uuid4()}.png'
    tcplot.plot_response(in_sig.clone(), output_sig.clone(), tag='Freq Response', save_path=plot_path)

    return in_path_to_file, output_sig.path_to_file, tc.detensor_dict(out_params_dict), plot_path

with gr.Blocks() as demo:
    gr.Markdown("### 💥 💥 💥 !!!!Text2FX (baseline demo) !!!!! 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")
    text = gr.Textbox(lines=5, label="i want this sound to be ...")
    criterion = gr.Radio(["standard", "directional_loss", "cosine-sim"], label="criterion", value="cosine-sim")
    params_init_type = gr.Radio(["random", "zeros"], label="initialization parameters type", value="zeros")
    num_iters = gr.Slider(0, 1000, value=600, step=25,label="number of iterations")
    fx_chain = gr.Dropdown(["reverb", "eq", "compression", "gain", "eq40"],  multiselect=True, label='Effects to include in FX chain', value=["eq"])

    process_button = gr.Button("Text2FX it!")
    in_compare_audio = gr.Audio(label="input sound", type="filepath")
    output_audio = gr.Audio(label="output sound", type="filepath")
    output_params = gr.JSON(label='output params')
    
    output_plot = gr.Image(label = "frequency response", type = "filepath")

    process_button.click(
        process_fn, 
        inputs={input_audio, text, criterion, num_iters, params_init_type, fx_chain}, 
        outputs={in_compare_audio, output_audio, output_params, output_plot}
    )

    
demo.launch(server_port=7869, share=True)
