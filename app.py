
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

ARTIFACTS_DIR = Path('/home/annie/research/text2fx/runs/app_artifacts')
shutil.rmtree(ARTIFACTS_DIR)
ARTIFACTS_DIR.mkdir()

def extract_label(k):
    parts = k.split('_')  # Split the string by '_'
    label = '_'.join(parts[-2:])  # Join the last two parts
    return label

def find_params(data):
    shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir()

    global channel
    # print(data[input_audio])
    # print(data[text])
    output_sig, out_params, out_params_dict = text2fx(
        model_name = 'ms_clap',
        sig = at.AudioSignal(data[input_audio], duration=3), 
        text=data[text],
        export_audio=True,
        channel=channel,
        criterion='cosine-sim',#data[criterion],
        params_init_type= 'random',
        n_iters= 50,
    )
    assert output_sig.path_to_file is not None

    in_path_to_file = os.path.join(os.path.dirname(output_sig.path_to_file), os.path.basename(output_sig.path_to_file).replace('_final', '_input'))

    # return in_path_to_file, output_sig.path_to_file, tc.detensor_dict(out_params_dict)
    detensor_params = tc.detensor_dict(out_params_dict)
    # print(detensor_params)
    params_out = []
    test_dict = {}
    for m in detensor_params:
        # print(m)
        param_dict = detensor_params[m]
        for k, value in param_dict.items():
            # print(k, value[0])
            params_out.append(value[0])
            test_dict[k] = value[0]
    # list_of_params = list(detensor_params.values())
    # print(list_of_params)

    output_sig.ensure_max_of_audio()
    assert output_sig.path_to_file is not None

    export_path = ARTIFACTS_DIR/f'{uuid.uuid4()}.wav'
    output_sig.write(export_path)
    return export_path, test_dict, *params_out

def apply_params(kwargs):
    shutil.rmtree(ARTIFACTS_DIR)
    ARTIFACTS_DIR.mkdir()
    global channel
    # print('==========KWARGS')
    # print(kwargs)

    new_params = {}    
    input_audio_path = kwargs.pop(input_audio)
    # print(input_audio_path)

    # print(kwargs)
    for k, v in kwargs.items():
        new_params[k.label] = v
        # print(new_params)

    new_params = {'ParametricEQ': new_params}
    print(new_params)
    params_dict = normalize_param_dict(new_params, channel)

    in_sig = at.AudioSignal(input_audio_path).resample(44_100).to_mono().ensure_max_of_audio()
    
    # depending on exact json dict output, this will change
    params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
    # if in_sig.batch_size != 1:
    #     params_list = params_list.transpose(0, 1)
    
    params = params_list.expand(in_sig.batch_size, -1) #shape = (n_batch, n_params)

    # print(params)
    out_sig = channel(in_sig.clone(), params)
    out_sig = out_sig.ensure_max_of_audio()

    export_path = ARTIFACTS_DIR/f'{uuid.uuid4()}.wav'
    out_sig.write(export_path)

    plot_path = ARTIFACTS_DIR/f'{uuid.uuid4()}.png'
    tcplot.plot_response(in_sig.clone(), out_sig.clone(), tag='Freq Response', save_path=plot_path)

    return export_path, plot_path

channel = tc.create_channel(['eq'])
                            
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("### 💥 💥 💥 Text2FX-EQ Interface 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")

    # ------ Run Text2FX to find optimal parameters ------
    # ==== setting up UI
    # -- no grouping
    text = gr.Textbox(lines=5, label="I want this sound to be ...")
    process_button = gr.Button("Find EQ parameters!")

    with gr.Row():
        output_audio_to_check = gr.Audio(label="Text2FX Params Preview", type="filepath")
        output_params = gr.JSON(label='Text2FX Params Preview params') #these are the output parameters

    #setting the sliders
    # (temporary) Output Audiosignal: apply EQ to params
    params_ui = {}
    for m in channel.modules:
        band_list = ["low_shelf", "band0", "band1", "band2", "band3", "high_shelf"]
        band_dicts = []
        for band in band_list:
            band_dict =  {k: v for k, v in m.param_ranges.items() if band in k}  
            band_dicts.append(band_dict)

        for band_dict, band in zip(band_dicts, band_list):
        # for band_dict in band_dicts:
            gr.Markdown(f"### {band}")
            with gr.Row():
                for k, range in band_dict.items(): 
                    param_type = extract_label(k)
                    if param_type == 'cutoff_freq':
                        scale = 2
                    else:
                        scale = 1
                    params_ui[k] = gr.Slider(
                        # label=f'{extract_label(k)} ({range[0]}, {range[1]})',
                        label = k,
                        minimum=range[0],
                        maximum=range[1],
                        value=(range[0] + range[1]) / 2,
                        info = f'Range: {range[0]}, {range[1]}',
                        scale=scale
                    )
  

    # ==== Actual process function to find params
    process_button.click(
        find_params, 
        inputs={input_audio, text},
        # outputs = set(params_ui.values()) | {output_audio_to_check, output_params} 
        outputs = {output_audio_to_check, output_params} | set(params_ui.values()) 

    )

    # ------ Apply Text2FX-parameters to original file ------
    # ==== Setting up UI
    # Output Audio File: apply EQ to params
    apply_button = gr.Button("Apply EQ parameters!")

    with gr.Row():
        output_audio = gr.Audio(label="output sound", type="filepath")
        output_plot = gr.Image(label = "frequency response", type = "filepath")
        # output_params = gr.JSON(label='output params') #these are the output parameters

    # ==== Actual process function to apply params
    apply_button.click(
        apply_params, 
        inputs={input_audio} | set(params_ui.values()),
        outputs={output_audio, output_plot}
    )
demo.launch(server_port=7863)

