
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

def generate_random_string():
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15)))
    return f'"{random_str}"'


def generate_random_nums():
    random_str = ''.join(random.choices(string.digits, k=random.randint(5, 15)))
    return f'"{random_str}"'


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
        # sig = at.AudioSignal(data[input_audio], duration=3), 
        sig = at.AudioSignal.salient_excerpt(data[input_audio], duration=3),
        text=data[text],
        export_audio=True,
        channel=channel,
        criterion='cosine-sim',#data[criterion],
        params_init_type= 'random',
        n_iters= 600,
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
    return tuple(params_out)
    # return export_path, test_dict, *params_out

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
        if k.label == "original":
            continue
        else:
            new_params[k.label] = v
        # print(new_params)

    new_params = {'ParametricEQ': new_params}
    print(new_params)
    params_dict = normalize_param_dict(new_params, channel)

    in_sig = at.AudioSignal(input_audio_path).resample(44_100).to_mono().ensure_max_of_audio()
    
    if kwargs[input_a] is not None:
        orig_sig = at.AudioSignal(kwargs[input_a])
    else:
        orig_sig = in_sig
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

    orig_export_path = ARTIFACTS_DIR/f'{uuid.uuid4()}.wav'
    orig_sig.write(orig_export_path)

    plot_path = ARTIFACTS_DIR/f'{uuid.uuid4()}.png'
    tcplot.plot_response(orig_sig.clone(), out_sig.clone(), tag='Freq Response', save_path=plot_path)

    return export_path, orig_export_path, plot_path

channel = tc.create_channel(['eq'])

sample_dict = {
    "speech": Path("/home/annie/research/text2fx/assets/text2eq/speech.wav"),
    "guitar": Path("/home/annie/research/text2fx/assets/text2eq/guitar.wav")
}
                 
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 💥 💥 💥 Text2EQ 💥 💥 💥")

    ## Initial UI for input audio and description text
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(label="a sound", type="filepath")
            feedback_text = gr.Markdown("")

        with gr.Column():
            text = gr.Textbox(lines=3, label="I want this sound to be ...")
        # with gr.Column():
    
    ## Button to process the audio and find EQ params
    process_button = gr.Button("Text2EQ It - Find EQ params!")

        # with gr.Column():
        #     output_audio_to_check = gr.Audio(label="Preview: Audio", type="filepath")
        #     output_params = gr.JSON(label='Preview: Params') #these are the output parameters

        # with gr.Column():
    # process_button = gr.Button("Text2FX - Find & Apply EQ params!")

    #setting the sliders
    # (temporary) Output Audiosignal: apply EQ to params
    gr.Markdown(f"### 6-Band Parametric EQ Controls")
    params_ui = {}

    for m in channel.modules:
        band_list = ["low_shelf", "band0", "band1", "band2", "band3", "high_shelf"]
        band_dicts = []
        for band in band_list:
            band_dict =  {k: v for k, v in m.param_ranges.items() if band in k}  
            band_dicts.append(band_dict)

        for band_dict, band in zip(band_dicts, band_list):
        # for band_dict in band_dicts:
            gr.Markdown(f"***{band}***")
            with gr.Row():
                for k, range in band_dict.items(): 
                    scale = 2 if extract_label(k) == 'cutoff_freq' else 1

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
        outputs = set(params_ui.values()) 
        #outputs = {output_audio_to_check, output_params} | set(params_ui.values()) 
    )

    # ------ Apply Text2FX-parameters to original file ------
    # ==== Setting up UI
    # Output Audio File: apply EQ to params
    apply_button = gr.Button("Apply EQ parameters!")
    # apply_feedback_text = gr.Markdown("")

    with gr.Row():
        with gr.Column():
            output_audio = gr.Audio(label="output sound", type="filepath", interactive=False)
            feedback_button = gr.Button("Use Output as New Input")
            # feedback_text = gr.Markdown("")
            input_a = gr.Audio(label='original', type="filepath", interactive=False)
        output_plot = gr.Image(label = "frequency response", type = "filepath")
        # output_params = gr.JSON(label='output params') #these are the output parameters


    # ==== Actual process function to apply params
    apply_button.click(
        apply_params, 
        inputs={input_audio} | set(params_ui.values()) | {input_a},
        outputs={output_audio, input_a, output_plot}
    )


    # ### Testing Feedback
    # Feedback button to feed output_audio back into input_audio

    # When feedback_button is clicked, set output_audio as the new input_audio
    feedback_button.click(
        lambda audio: (audio, "loaded output as input! // " + generate_random_string()),  # return the output_audio as input
        inputs=output_audio,
        outputs=[input_audio, feedback_text]
    )

demo.launch(server_port=7865, share=True)

