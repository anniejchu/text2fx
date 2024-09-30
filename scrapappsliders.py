import audiotools as at
import gradio as gr
import numpy as np

from text2fx.scrap import text2fx_paper as text2fxrun
import text2fx.core as tc
import torch
import os

def process_fn(data):
    print(data[input_audio])
    print(data[text])
    print(data[fx_chain])
    
    channel = tc.create_channel(data[fx_chain])
    output_sig, out_params, out_params_dict = text2fxrun(
        model_name='ms_clap',
        sig=at.AudioSignal(data[input_audio]), 
        text=data[text],
        export_audio=True,
        channel=channel,
        criterion='directional_loss',
        params_init_type='random',
        n_iters=100,
    )
    
    assert output_sig.path_to_file is not None

    in_path_to_file = os.path.join(
        os.path.dirname(output_sig.path_to_file), 
        os.path.basename(output_sig.path_to_file).replace('_final', '_input')
    )

    # Return the input file path, output file path, and params dict
    return (
        in_path_to_file,
        output_sig.path_to_file,
        out_params_dict
    )

def create_sliders(params):
    sliders = []
    for key, value in params.items():
        slider = gr.Slider(minimum=0, maximum=1, value=value, step=0.01, label=key)
        sliders.append(slider)
    return sliders

with gr.Blocks() as demo:
    gr.Markdown("### 💥 💥 💥 !!!!text2fx (baseline demo) !!!!! 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")
    text = gr.Textbox(lines=5, label="I want this sound to be ...")
    fx_chain = gr.Dropdown(
        ["reverb", "eq", "compressor", "gain", "eq40"], 
        multiselect=True, 
        label='Effects to include in FX chain', 
        value=["eq"]
    )

    process_button = gr.Button("Hit it, let's go!")
    in_compare_audio = gr.Audio(label="Input sound", type="filepath")
    output_audio = gr.Audio(label="Output sound", type="filepath")
    
    # Create a dynamic area for sliders
    slider_group = gr.Group(label='Output Params Sliders')

    process_button.click(
        process_fn, 
        inputs={input_audio, text, fx_chain}, 
        outputs={in_compare_audio, output_audio, slider_group}
    )

    # Create a placeholder to store the sliders
    sliders_output = gr.State()

    def update_sliders(params):
        sliders = create_sliders(params)
        return sliders

    # Update sliders after processing
    process_button.click(
        lambda params: update_sliders(params), 
        inputs=slider_group,  # Input the parameters to update sliders
        outputs=slider_group
    )

demo.launch(server_port=7863, share=True)
