
import audiotools as at
import gradio as gr
import numpy as np

from text2fx.text2fx_app import text2fx_paper as text2fx
import text2fx.core as tc
import text2fx.core_plotting as tcplot
import torch
import os
from process_file_from_params import normalize_param_dict

def find_params(data):
    global channel
    print(data[input_audio])
    print(data[text])
    # print(data[criterion])
    # print(data[fx_chain])
    # channel = tc.create_channel(data[fx_chain])
    # breakpoint()
    output_sig, out_params, out_params_dict = text2fx(
        model_name = 'ms_clap',
        sig = at.AudioSignal(data[input_audio], duration=3), 
        text=data[text],
        export_audio=True,
        channel=channel,
        criterion='cosine-sim',#data[criterion],
        params_init_type= 'random',
        n_iters= 600,
        # save_dir='/tmp/gradio/f6bc142ba5459e2e8db732b3665face1592bdaa2/'
    )
    assert output_sig.path_to_file is not None

    in_path_to_file = os.path.join(os.path.dirname(output_sig.path_to_file), os.path.basename(output_sig.path_to_file).replace('_final', '_input'))

    # return in_path_to_file, output_sig.path_to_file, tc.detensor_dict(out_params_dict)
    detensor_params = tc.detensor_dict(out_params_dict)
    print(detensor_params)
    params_out = []
    for m in detensor_params:
        print(m)
        param_dict = detensor_params[m]
        for k, value in param_dict.items():
            print(k, value[0])
            params_out.append(value[0])

    # list_of_params = list(detensor_params.values())
    # print(list_of_params)

    output_sig.ensure_max_of_audio()
    assert output_sig.path_to_file is not None
    output_sig.write(output_sig.path_to_file)
    return *params_out, output_sig.path_to_file

def apply_params(kwargs):
    global channel
    print('KWARGS')
    print(kwargs)

    new_params = {}    
    input_audio_path = kwargs.pop(input_audio)
    # print(input_audio_path)

    print(kwargs)
    for k, v in kwargs.items():
        new_params[k.label] = v
        # print(new_params)

    new_params = {'ParametricEQ': new_params}
    params_dict = normalize_param_dict(new_params, channel)

    print(params_dict)
    # params_dict = normalize_param_dict(_params_dict, fx_channel) #normalizing 
    in_sig = at.AudioSignal(input_audio_path).resample(44_100).to_mono().ensure_max_of_audio()
    # depending on exact json dict output, this will change
    params_list = torch.tensor([value for effect_params in params_dict.values() for value in effect_params.values()])
    # if in_sig.batch_size != 1:
    #     params_list = params_list.transpose(0, 1)
    params = params_list.expand(in_sig.batch_size, -1) #shape = (n_batch, n_params)

    print(params)
    out_sig = channel(in_sig.clone(), params)
    out_sig = out_sig.ensure_max_of_audio()

    assert out_sig.path_to_file is not None

    out_sig.write(out_sig.path_to_file)


    #TODO: add plot_response image

    return out_sig.path_to_file #out_path


channel = tc.create_channel(['eq'])
                            
with gr.Blocks() as demo:
    gr.Markdown("### 💥 💥 💥 Text2FX-EQ Interface 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")
    text = gr.Textbox(lines=5, label="I want this sound to be ...")

    # criterion = gr.Radio(["standard", "directional_loss", "cosine-sim"], label="criterion", value="cosine-sim")
    # params_init_type = gr.Radio(["random", "zeros"], label="initialization parameters type", value="zeros")
    # num_iters = gr.Slider(0, 1000, value=600, step=25,label="number of iterations")
    # fx_chain = gr.Dropdown(["reverb", "eq", "compressor", "gain", "eq40"],  multiselect=True, label='Effects to include in FX chain', value=["eq"])


    # ------ Run Text2FX to find optimal parameters ------
    # ==== setting up UI
    process_button = gr.Button("Find EQ parameters!")
    # EQ sliders, TODO: vertical mode
    params_ui = {}
    for m in channel.modules:
        for k, range in m.param_ranges.items():
            print(range[0], range[1])
            params_ui[k] = gr.Slider(label = k, minimum = range[0], maximum = range[1], value = (range[0]+range[1])/2)

        print(m)
        print(m.param_ranges)

    # (temporary) Output Audiosignal: apply EQ to params
    # output_audio_to_check = gr.Audio(label="output to check", type="filepath")

    # ==== Actual process function to find params
    process_button.click(
        find_params, 
        inputs={input_audio, text},#, fx_chain}, 
        outputs = set(params_ui.values())# | {output_audio_to_check}
    )

    # ------ Apply Text2FX-parameters to original file ------
    # ==== Setting up UI
    # Output Audio File: apply EQ to params
    apply_button = gr.Button("Apply EQ parameters!")

    output_audio = gr.Audio(label="output sound", type="filepath")
    # output_params = gr.JSON(label='output params') #these are the output parameters

    # ==== Actual process function to apply params
    apply_button.click(
        apply_params, 
        inputs={input_audio} | set(params_ui.values()),
        outputs={output_audio} #TODO: add image here?
    )


    
demo.launch(server_port=7863)

