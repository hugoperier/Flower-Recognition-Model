#!/bin/bash 
#PBS -l nodes=1:gpus=1:ppn=1

source activate tf15

cd /home/brunel_m/Flower-Recognition-Model
pwd
python3 main.py

conda deactivate