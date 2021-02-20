#!/bin/bash

MAIN_MODULE=$HOME/scratch/gantavya/chacha/MTSG/

JOB_DIR=$MAIN_MODULE/narma_1/B
cd $JOB_DIR

source activate apricot_old

python run.py narma narma_2d_1_n0.csv 0 B 0 0 10 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 10 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 15 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 15 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 20 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 20 1 1 
python run.py narma narma_2d_1_n0.csv 0 B 0 0 25 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 25 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 30 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 30 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 35 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 35 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 40 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 40 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 45 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 45 1 1
python run.py narma narma_2d_1_n0.csv 0 B 0 0 42 1 0
python run.py narma narma_2d_1_n0.csv 0 B 0 0 42 1 1

