import subprocess
from text2fx.core import RUNS_DIR, ASSETS_DIR

# Initializing some start parameteres
base_save_dir = RUNS_DIR / 'test625_wordtargets' #change this at will
# criterion = "cosine-sim"
n_iters = 600
lr = 0.01

# Setting up DEFAULT text targets
top10_eq = ["warm", "cold", "soft", "loud", "happy", "bright", "soothing", "harsh", "heavy", "cool"]
prefix = 'this sound is '
y = [prefix + x for x in top10_eq]


#625: EXPERIMENTING WITH OTHER WORDS
def experiment_params_random(text_targets, model_name: str, input_audio: str, criterion:str = 'cosine-sim', roll: str = 'all', roll_amt:int = 0):
    for text in text_targets:
        for i in range(2):
            print(f'starting random seed {i} for {text}')
            command = [
                "python", "-m", "text2fx",
                "--model_name", model_name,
                "--input_audio", input_audio,
                "--text", text,
                "--criterion", criterion,
                "--n_iters", str(n_iters),
                "--lr", str(lr),
                "--save_dir", str(base_save_dir/ model_name / input_audio.stem / text.replace(prefix, "").replace(" ", "_") / f'seed{i}_{criterion}'),
                "--params_init_type", "random",
                "--seed_i", str(i),
                # "--roll", str(roll),
                # "--roll_amt", str(roll_amt)
            ]
            subprocess.run(command)

def experiment_params_zeros(text_targets, model_name: str, input_audio: str, criterion:str = 'cosine-sim', roll: str = 'all', roll_amt:int = 0):
    for text in text_targets:
        print(f'starting zeros for {text}')
        command = [
            "python", "-m", "text2fx",
            "--model_name", model_name,
            "--input_audio", input_audio,
            "--text", text,
            "--criterion", criterion,
            "--n_iters", str(n_iters),
            "--lr", str(lr),
            "--save_dir", str(base_save_dir/ model_name / input_audio.stem / text.replace(prefix, "").replace(" ", "_") /f'zeros_{criterion}'),
            "--params_init_type", "zeros",
            # "--seed_i", str(i),
            # "--roll", str(roll),
            # "--roll_amt", str(roll_amt)
        ]
        subprocess.run(command)

#torch_roll_test



# **** Choosing model 
        ## NEED TO SWITCH TRANSFORMER VERSIONS
        #laion-clap = transformers==4.30.0
        #msclap = transformers>=4.34.0, try transformers==4.40.0
model_name ='ms_clap'# "laion_clap" #"ms_clap"# 


words_lite = ['bright', 'warm', 'heavy', 'soft']
comparatives_lite = ['brighter', 'warmer', 'heavier', 'softer']
emphasis_lite = ['very bright', 'very warm', 'very heavy', 'very soft']
# subdue_lite = ['less bright', 'less warm', 'less heavy', 'less soft']
words_xlite = ['heavy']

text_targets_lite = [prefix + x for x in words_lite]
text_targets_xlite = [prefix + x for x in words_xlite]

comp_targets_lite = [prefix + x for x in comparatives_lite]
emphasis_targets_lite = [prefix + x for x in emphasis_lite]
# subdue_targets_lite = [prefix + x for x in subdue_lite]


# input_audio_files = ['bass_idmt_007', 'drums_beatles_musdb', 'vocal_beatles_musdb', 'guitar_idmt_ibanez_rock_4_110BPM']
input_audio_files = ['drums_beatles_musdb']

# running thru different word prompts
for a in input_audio_files:
    input_audio = ASSETS_DIR / 'multistem_examples' / f'{a}.wav' #path    

    # experiment_params_random(text_targets_lite, model_name, input_audio, 'cosine-sim')
    # experiment_params_zeros(text_targets_lite, model_name, input_audio, 'cosine-sim')

    # experiment_params_random(comp_targets_lite, model_name, input_audio, 'cosine-sim')
    # experiment_params_zeros(comp_targets_lite, model_name, input_audio, 'cosine-sim')

    experiment_params_random(text_targets_xlite, model_name, input_audio, 'directional_loss')
    experiment_params_zeros(text_targets_xlite, model_name, input_audio, 'directional_loss')

    # experiment_params_random(emphasis_targets_lite, model_name, input_audio, 'cosine-sim')
    # experiment_params_zeros(emphasis_targets_lite, model_name, input_audio, 'cosine-sim')

    # experiment_params_random(subdue_targets_lite, model_name, input_audio, 'cosine-sim')
    # experiment_params_zeros(subdue_targets_lite, model_name, input_audio, 'cosine-sim')



# run_it_zero(y, model_name, input_audio)




