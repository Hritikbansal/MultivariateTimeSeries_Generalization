#!/bin/bash

MAIN_MODULE=$HOME/scratch/gantavya/chacha/MTSG/

JOB_DIR=$MAIN_MODULE/narma_1/B
cd $JOB_DIR

source activate apricot_old

python run.py narma narma_2d_1_n0.csv 0 B 500 0 10
python run.py narma narma_2d_1_n0.csv 0 B 0 1 10
python run.py narma narma_2d_1_n0.csv 0 B 500 0 15
python run.py narma narma_2d_1_n0.csv 0 B 0 1 15
python run.py narma narma_2d_1_n0.csv 0 B 500 0 20
python run.py narma narma_2d_1_n0.csv 0 B 0 1 20
python run.py narma narma_2d_1_n0.csv 0 B 500 0 25
python run.py narma narma_2d_1_n0.csv 0 B 0 1 25
python run.py narma narma_2d_1_n0.csv 0 B 500 0 30
python run.py narma narma_2d_1_n0.csv 0 B 0 1 30
python run.py narma narma_2d_1_n0.csv 0 B 500 0 35
python run.py narma narma_2d_1_n0.csv 0 B 0 1 35
python run.py narma narma_2d_1_n0.csv 0 B 500 0 40
python run.py narma narma_2d_1_n0.csv 0 B 0 1 40
python run.py narma narma_2d_1_n0.csv 0 B 500 0 45
python run.py narma narma_2d_1_n0.csv 0 B 0 1 45
python run.py narma narma_2d_1_n0.csv 0 B 500 0 42
python run.py narma narma_2d_1_n0.csv 0 B 0 1 42

