
import audiotools as at
import gradio as gr
import numpy as np

from text2fx.__main__ import text2fx
import text2fx.core as tc

fx_channel = tc.create_channel(['reverb', 'eq'])

def process_fn(data):
    print(data[input_audio])
    print(data[text])
    print(data[criterion])
    channel = fx_channel
    output_sig, out_params_dict = text2fx(
        model_name = 'ms_clap',
        sig = at.AudioSignal(data[input_audio]), 
        text=data[text],
        export_audio=True,
        channel=channel,
        criterion=data[criterion],
        params_init_type='random',
        n_iters=100,
        # save_dir='/tmp/gradio/f6bc142ba5459e2e8db732b3665face1592bdaa2/'
    )
    assert output_sig.path_to_file is not None
    return output_sig.path_to_file, out_params_dict

with gr.Blocks() as demo:
    gr.Markdown("### 💥 💥 💥 !!!!text 2 fx!!!!! 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")
    text = gr.Textbox(lines=5, label="i want this sound to be ...")
    criterion = gr.Radio(["standard", "directional_loss", "cosine-sim"], label="criterion", value="cosine-sim")

    process_button = gr.Button("hit it lets go!")
    output_audio = gr.Audio(label="output sound", type="filepath")
    output_params = gr.JSON(label='output params')
    process_button.click(
        process_fn, 
        inputs={input_audio, text, criterion}, 
        
        #TODO: add output_params to outputs
        outputs={output_audio, output_params}
    )

    
demo.launch(server_port=7861)