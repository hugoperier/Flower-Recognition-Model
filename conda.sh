#!/bin/bash 
#PBS -l nodes=1:gpus=1:ppn=1

source activate tf15

cd /home/brunel_m/Flower-Recognition-Model
python3 no_dropout.py
python3 train_dropout.py
python3 finetunning.py

conda deactivate