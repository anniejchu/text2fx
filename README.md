<h1 align="center">Text2FX</h1>

This repository contains utilities for mapping text descriptions to audio effect parameters.

## Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects
Read the paper [here](https://arxiv.org/abs/2409.18847)! Accepted at ICASSP 2025

**Abstract**
This work introduces Text2FX, a method that leverages CLAP embeddings and differentiable digital signal processing to control audio effects, such as equalization and reverberation, using open-vocabulary natural language prompts (e.g., "make this sound in-your-face and bold"). Text2FX operates without retraining any models, relying instead on single-instance optimization within the existing embedding space, thus enabling a flexible, scalable approach to open-vocabulary sound transformations through interpretable and disentangled FX manipulation. We show that CLAP encodes valuable information for controlling audio effects and propose two optimization approaches using CLAP to map text to audio effect parameters. While we demonstrate with CLAP, this approach is applicable to any shared text-audio embedding space. Similarly, while we demonstrate with equalization and reverberation, any differentiable audio effect may be controlled.  %to control EQ and Reverb. 
We conduct a listener study with diverse text prompts and source audio to evaluate the quality and alignment of these methods with human perception. 
<!-- ## Contents
  * <a href="#install">Installation</a>
  * <a href="#cli"> Text2FX via CLI</a>
  * <a href="#demo">Text2FX Demo</a>
   * <a href="#citations">Cite</a>
 -->


<h2 id="install">Installation</h2>

1. Create conda environment with Python 3.9:
   ```
   conda create -y -n text2fx python=3.9
   source activate text2fx
   ```

   If you want to use Jupyter, run the following to add your conda environment as a kernel:
   ```
   conda install -y -c conda-forge jupyterlab
   conda install -y -c anaconda ipykernel
   python -m ipykernel install --user --name=text2fx
   ```

2. Clone repository:
   ```
   git clone https://github.com/anniejchu/text2fx.git
   pip install -e ./text2fx

   ```

3. Install dependencies:
   ```
   cd text2fx
   python -m pip install -r requirements.txt
   ```
<!-- 
## Running the Gradio UI

1. Run the following command to start the Gradio UI:
```
python app.py
``` -->

<h2 id="cli">Running Text2FX via command line</h2>
Run the following command to start the Text2FX command line interface:

### Quick Use: 1 audio file, 1 text descriptor
```
python -m text2fx.apply assets/multistem_examples/10s/bass.wav eq 'warm like a hug' \
    --export_dir experiments/prod_final \
    --learning_rate 0.01 \
    --params_init_type random \
    --n_iters 600 \
    --criterion cosine-sim \
```

### Batching: n audio files AND/OR n text descriptors
**Case 1: multiple audio files, single text_target**
```
python -m text2fx.applybatch \
    --audio_source assets/multistem_examples/10s \
    --descriptions_source "cold" \
    --fx_chain eq \
    --export_dir experiments/prod_final/case1
```
**Case 2: single audio file, multiple text_target**
```
python -m text2fx.applybatch \
    --audio_source assets/multistem_examples/10s/drums.wav \
    --descriptions_source "cold, warm, like a trumpet, muffled, lonely like a ghost" \
    --fx_chain eq compression\
    --export_dir experiments/prod_final/case2
```
**Case 3:  multiple audio files, multiple text_targets (must have same # of files to targets)**
```
python -m text2fx.applybatch \
    --audio_source assets/multistem_examples/10s \
    --descriptions_source "cold, warm, like a trumpet, muffled, lonely like a ghost" \
    --fx_chain eq reverb \
    --export_dir experiments/prod_final/case3
```

<h2 id="demo">Text2FX Demo</h2>
 Try Text2FX yourself: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anniejchu/text2fx/blob/clean/notebooks/demo.ipynb)


<h2 id="citations">Citation</h2>
If you use Text2FX, please cite via

```bibtex
@misc{chu2024text2fxharnessingclapembeddings,
      title={Text2FX: Harnessing CLAP Embeddings for Text-Guided Audio Effects}, 
      author={Annie Chu and Patrick O'Reilly and Julia Barnett and Bryan Pardo},
      year={2024},
      eprint={2409.18847},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2409.18847}, 
}
```   