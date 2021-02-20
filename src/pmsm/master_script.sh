#!/bin/bash

MAIN_MODULE=$HOME/scratch/gantavya/chacha/MTSG/gannu

JOB_DIR=$MAIN_MODULE/PMSM_B/cont_v2/new
cd $JOB_DIR

source activate apricot_old


python run.py PMSM PMSM.csv 0 B 0 0 10
python run.py PMSM PMSM.csv 0 B 0 1 10
python run.py PMSM PMSM.csv 0 B 0 0 15
python run.py PMSM PMSM.csv 0 B 0 1 15
python run.py PMSM PMSM.csv 0 B 0 0 20
python run.py PMSM PMSM.csv 0 B 0 1 20
python run.py PMSM PMSM.csv 0 B 0 0 25
python run.py PMSM PMSM.csv 0 B 0 1 25
python run.py PMSM PMSM.csv 0 B 0 0 30
python run.py PMSM PMSM.csv 0 B 0 1 30
python run.py PMSM PMSM.csv 0 B 0 0 35
python run.py PMSM PMSM.csv 0 B 0 1 35
python run.py PMSM PMSM.csv 0 B 0 0 40
python run.py PMSM PMSM.csv 0 B 0 1 40
python run.py PMSM PMSM.csv 0 B 0 0 42
python run.py PMSM PMSM.csv 0 B 0 1 42
python run.py PMSM PMSM.csv 0 B 0 0 45
python run.py PMSM PMSM.csv 0 B 0 1 45
