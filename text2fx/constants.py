from pathlib import Path
import torch 

PROJECT_DIR = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_DIR / "assets"
PRETRAINED_DIR = PROJECT_DIR / "pretrained"
DATA_DIR = PROJECT_DIR / "data"
RUNS_DIR = PROJECT_DIR / "runs"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"

# # setting sample rate
SAMPLE_RATE = 44_100  # NOTE: should this be here? clap take something else?
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" #


EQ_freq_bands = [20, 50, 83, 120, 161, 208, 259, 318, 383, 455, 537, 628, 729, 843, 971, 
              1114, 1273, 1452, 1652, 1875, 2126, 2406, 2719, 3070, 3462, 3901, 
              4392, 4941, 5556, 6244, 7014, 7875, 8839, 9917, 11124, 12474, 13984, 
              15675, 17566, 19682]

EQ_words_top_10 = ["warm", "cold", "soft", "loud", "happy", "bright", "soothing", "harsh", "heavy", "cool"]
