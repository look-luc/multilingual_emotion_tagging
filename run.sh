#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j.log
#SBATCH --job-name=multilang_emo
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

export HF_HOME="/projects/$USER/.cache/huggingface"
mkdir -p $HF_HOME

set -uo pipefail

REPO_ROOT="/projects/$USER"
cd "$REPO_ROOT"

module purge
module load anaconda
module load cuda/12.1.1
#module load ffmpeg

set +u && conda activate multilingual_emotion_tagging && set -u

cd "$REPO_ROOT/multilingual_emotion_tagging"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$(python3 -m site --user-site | sed 's/python.*/nvidia\/npp\/lib/'):$LD_LIBRARY_PATH
python3 main.py