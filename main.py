import json

import gradio as gr

from audio_properties import get_audio_properties


def sound_inspect(audio_path):
    props = get_audio_properties(audio_path, get_bpm=True)
    return [gr.Number(v, label=k) for k, v in props.items()]


with gr.Blocks() as demo:
    gr.Markdown(
        """Sound Inspector"""
    )
    inp = gr.Audio(label=f"Input audio", type='filepath')
    out = []
    with gr.Row():
        out += [gr.Number(), gr.Number(), gr.Number()]
    with gr.Row():
        out += [gr.Number(), gr.Number(), gr.Number()]
    with gr.Row():
        out += [gr.Number()]
    inp.change(sound_inspect, inp, out)

demo.launch()
