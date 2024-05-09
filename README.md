<h1 align="center">Text2FX</h1>

This repository contains utilities for mapping text descriptions to audio effect parameters.

## Contents
  * <a href="#install">Installation</a>

<h2 id="install">Installation</h2>


1. Install MiniConda if necessary (this requires a shell restart). You can check whether you already have an installation with the shell command `conda`. If you get an error, run:
   ```
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

2. Create conda environment:
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

3. Clone repository:
   ```
   git clone https://github.com/anniejchu/text2fx.git
   ```

4. Install dependencies:
   ```
   cd text2fx
   python -m pip install -r requirements.txt
   ```

5. Grant permissions for scripts:
   ```
   chmod -R u+x scripts/
   ```

6. Download datasets (226G total) to a specified directory `<DATA-DIR>`:
   ```
   ./scripts/setup/download_data.sh <DATA-DIR>
   ```
   The script also creates a symbolic link between `text2fx/data/` and `<DATA-DIR>` so that the data is accessible within the project. Depending on your internet connection, the download could take a few hours, so you may want to run within a `tmux` session.

## Running the Gradio UI

1. Run the following command to start the Gradio UI:
```
python app.py
```

## Running txt2fx via command line

1. Run the following command to start the txt2fx command line interface:
```
python text2fx.py --input_audio "assets/speech_examples/VCTK_p225_001_mic1.flac"\
                 --text "this sound is dark, boomy and distored" \
                 --criterion "cosine-sim" \
                 --n_iters 500 \
                 --lr 0.01 

```
