#!/bin/bash 
#PBS -l nodes=1:gpus=1:ppn=1

source activate keras_tf1.15

cd /home/perier_h/Documents/finals
python3 main.py

conda deactivate
