import subprocess
from text2fx.core import RUNS_DIR, ASSETS_DIR

#running it with params set to random based on deterministic seed, then pushed up to sigmoid
def run_it(text_targets, model_name: str, input_audio: str):
    for text in text_targets:
        for i in range(3):
            print(f'starting random, seed {i}')
            command = [
                "python", "-m", "text2fx",
                "--model_name", model_name,
                "--input_audio", input_audio,
                "--text", text,
                "--criterion", criterion,
                "--n_iters", str(n_iters),
                "--lr", str(lr),
                "--save_dir", str(base_save_dir/ model_name / audio_type / text.replace(prefix, "").replace(" ", "_") / f'seed{i}_0'),
                "--params_init_type", "random",
                "--seed_i", str(i),


            ]
            subprocess.run(command)

#running it with params set to 0, then pushed up to sigmoid
def run_it_zero(text_targets, model_name: str, input_audio: str):
    for text in text_targets:
        for i in range(3):
            print(f'starting zeros, round {i}')
            command = [
                "python", "-m", "text2fx",
                "--model_name", model_name,
                "--input_audio", input_audio,
                "--text", text,
                "--criterion", criterion,
                "--n_iters", str(n_iters),
                "--lr", str(lr),
                "--save_dir", str(base_save_dir/ model_name / audio_type / text.replace(prefix, "").replace(" ", "_") / f'zeros_{i}'),
                "--params_init_type", "zeros",
            ]
            subprocess.run(command)



#616: EXPERIMENTING WITH TORCH.ROLL
def torchroll_test(text_targets, model_name: str, input_audio: str, roll: str, roll_amt:int = 0):
    for text in text_targets:
        for i in range(2):
            print(f'starting random, seed {i}')
            command = [
                "python", "-m", "text2fx",
                "--model_name", model_name,
                "--input_audio", input_audio,
                "--text", text,
                "--criterion", criterion,
                "--n_iters", str(n_iters),
                "--lr", str(lr),
                "--save_dir", str(base_save_dir/ model_name / input_audio.stem / text.replace(prefix, "").replace(" ", "_") /f'seed{i}_0'/ f'roll_{roll}_{roll_amt}'),
                "--params_init_type", "random",
                "--seed_i", str(i),
                "--roll", str(roll),
                "--roll_amt", str(roll_amt)
            ]
            subprocess.run(command)

#torch_roll_test

# Setting up text targets
top10_eq = ["warm", "cold", "soft", "loud", "happy", "bright", "soothing", "harsh", "heavy", "cool"]
prefix = 'this sound is '
y = [prefix + x for x in top10_eq]

# Setting up audio file
# audio_type = "bass_idmt_007"#"drums_beatles_musdb"
# input_audio = ASSETS_DIR / 'multistem_examples' / f'{audio_type}.wav' #path

# Initializing some start parameteres
base_save_dir = RUNS_DIR / 'test616_torchroll' #change this at will
criterion = "cosine-sim"
n_iters = 600
lr = 0.01

# **** Choosing model 
        ## NEED TO SWITCH TRANSFORMER VERSIONS
        #laion-clap = transformers==4.30.0
        #msclap = transformers>=4.34.0, try transformers==4.40.0
model_name ='laion_clap'# "laion_clap" #"ms_clap"# 
words_lite = ['bright', 'warm']
input_audio_files = ['bass_idmt_007', 'drums_beatles_musdb']
for a in input_audio_files:
    input_audio = ASSETS_DIR / 'multistem_examples' / f'{a}.wav' #path
    # print(input_audio.stem, input_audio.exists())
    torchroll_test(words_lite, model_name, input_audio, 'none')
    torchroll_test(words_lite, model_name, input_audio, 'all')
    torchroll_test(words_lite, model_name, input_audio, 'custom', 1000)

# run_it_zero(y, model_name, input_audio)




