
import audiotools as at
import gradio as gr
import numpy as np

import text2fx


def process_fn(data):
    print(data[input_audio])
    channel = text2fx.get_default_channel()
    output_sig = text2fx.text2fx(
        at.AudioSignal(data[input_audio]), 
        text=data[text],
        channel=channel,
    )
    assert output_sig.path_to_file is not None
    return output_sig.path_to_file

with gr.Blocks() as demo:
    gr.Markdown("### 💥 💥 💥 !!!!text 2 fx!!!!! 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")
    text = gr.Textbox(lines=5, label="how shall we transform this sound?")
    criterion = gr.Radio(["standard", "directional_loss", "cosine-sim"], label="criterion", value="cosine-sim")

    process_button = gr.Button("hit it lets go!")
    output_audio = gr.Audio(label="output sound", type="filepath")

    process_button.click(
        process_fn, 
        inputs={input_audio, text}, 
        outputs={output_audio}
    )

    
demo.launch()