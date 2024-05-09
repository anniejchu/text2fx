
import audiotools as at
import gradio as gr

import text2fx


def process_fn(data):
    print(data[input_audio])
    channel = text2fx.get_default_channel()
    output_sig = text2fx.text2fx(
        at.AudioSignal(data[input_audio]), 
        text=data[text],
        channel=channel,
    )
    return output_sig.sample_rate, output_sig.samples[0].cpu().numpy()

with gr.Blocks() as demo:
    gr.Markdown("### 💥 💥 💥 !!!!text 2 fx!!!!! 💥 💥 💥")
    input_audio = gr.Audio(label="a sound", type="filepath")
    text = gr.Textbox(lines=5, label="how shall we transform this sound?")
    criterion = gr.Radio(["standard", "directional_loss", "cosine-sim"], label="criterion", value="standard")

    process_button = gr.Button("hit it lets go!")
    output_audio = gr.Audio(label="output sound")

    process_button.click(
        process_fn, 
        inputs={input_audio, text}, 
        outputs={output_audio}
    )

    
demo.launch()