#!/bin/bash

set -e

################################################################################
# Environment
################################################################################

# Constants
SETUP_SCRIPTS_DIR=$(eval dirname "$(readlink -f "$0")")
SCRIPTS_DIR="$(dirname "$SETUP_SCRIPTS_DIR")"
PROJECT_DIR="$(dirname "$SCRIPTS_DIR")"
LOCAL_DATA_DIR="${PROJECT_DIR}/data/"

echo "SETUP SCRIPTS"
echo $SETUP_SCRIPTS_DIR

if [ "$1" == "" ]; then
  echo "Usage: $0 <DATA-DIR>"
  exit
fi

DATA_DIR=$(echo $1 | sed 's:/*$::')

# echo Creating symlink from "$LOCAL_DATA_DIR" to "$DATA_DIR"
# # if u w
# ln -s "$DATA_DIR" "$(dirname $LOCAL_DATA_DIR)"


################################################################################
# Download data
################################################################################

# MTG-Jamendo (152G)
# python "$SETUP_SCRIPTS_DIR"/download_mtg_jamendo.py "$DATA_DIR"


# MUSDB18-HQ (23G)
echo "downloading MUSDB18-HQ..."
wget -O "$DATA_DIR"/musdb18hq.zip \
  https://zenodo.org/records/3338373/files/musdb18hq.zip
unzip "$DATA_DIR"/musdb18hq.zip -d "$DATA_DIR"/musdb_hq
rm -f "$DATA_DIR"/musdb18hq.zip


# VCTK (12G)
echo "downloading VCTK..."
wget -O "$DATA_DIR"/DS_10283_3443.zip \
  https://datashare.ed.ac.uk/download/DS_10283_3443.zip
unzip "$DATA_DIR"/DS_10283_3443.zip -d "$DATA_DIR"/VCTK
rm -f "$DATA_DIR"/DS_10283_3443.zip
unzip "$DATA_DIR"/VCTK/VCTK-Corpus-0.92.zip -d "$DATA_DIR"/VCTK/
rm -f "$DATA_DIR"/VCTK/VCTK-Corpus-0.92.zip
rm -rf "$DATA_DIR"/VCTK/wav48_silence_trimmed/p315  # Speaker p315 is missing text data


# DAPS (21G)
echo "downloading DAPS..."
wget -O "$DATA_DIR"/daps.tar.gz \
  https://zenodo.org/record/4660670/files/daps.tar.gz?download=1
tar -xzvf "$DATA_DIR"/daps.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/daps.tar.gz
find "$DATA_DIR"/daps -name "._*" -type f -delete  # Remove macOS junk files


# LibriTTS-R (81G)
echo "downloading LibriTTS-R test_clean..."
wget -O "$DATA_DIR"/test_clean.tar.gz \
 https://www.openslr.org/resources/141/test_clean.tar.gz
tar -xzvf "$DATA_DIR"/test_clean.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/test_clean.tar.gz

echo "downloading LibriTTS-R train_clean_100..."
wget -O "$DATA_DIR"/train_clean_100.tar.gz \
 https://www.openslr.org/resources/141/train_clean_100.tar.gz
tar -xzvf "$DATA_DIR"/train_clean_100.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/train_clean_100.tar.gz

echo "downloading LibriTTS-R train_clean_360..."
wget -O "$DATA_DIR"/train_clean_360.tar.gz \
 https://www.openslr.org/resources/141/train_clean_360.tar.gz
tar -xzvf "$DATA_DIR"/train_clean_360.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/train_clean_360.tar.gz

echo "downloading LibriTTS-R test_other..."
wget -O "$DATA_DIR"/test_other.tar.gz \
 https://www.openslr.org/resources/141/test_other.tar.gz
tar -xzvf "$DATA_DIR"/test_other.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/test_other.tar.gz

echo "downloading LibriTTS-R train_other_500..."
wget -O "$DATA_DIR"/train_other_500.tar.gz \
 https://www.openslr.org/resources/141/train_other_500.tar.gz
tar -xzvf "$DATA_DIR"/train_other_500.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/train_other_500.tar.gz

echo "downloading LibriTTS-R dev_clean..."
wget -O "$DATA_DIR"/dev_clean.tar.gz \
 https://www.openslr.org/resources/141/dev_clean.tar.gz
tar -xzvf "$DATA_DIR"/dev_clean.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/dev_clean.tar.gz

echo "downloading LibriTTS-R dev_other..."
wget -O "$DATA_DIR"/dev_other.tar.gz \
 https://www.openslr.org/resources/141/dev_other.tar.gz
tar -xzvf "$DATA_DIR"/dev_other.tar.gz -C "$DATA_DIR"
rm -f "$DATA_DIR"/dev_other.tar.gz