# MultivariateTimeSeries_Generalization

This repository is an official pytorch implementation of the paper - **Systematic Generalization for Prediction Control in MultiVariate Time Series** ([arxiv](https://arxiv.org/abs/2102.05602)).

## Requirements

- python 3
- pytorch >= 1.1
- numpy 
- matplotlib 

You can use anaconda (python library manager) or pip. We used anaconda, and downloaded all the dependencies in an anaconda environment.

## Authors
1. [Hritik Bansal](https://sites.google.com/view/hbansal)
2. [Gantavya Bhatt](https://sites.google.com/view/gbhatt/)
3. [Pankaj Malhotra](https://scholar.google.co.in/citations?user=HP4M0MkAAAAJ&hl=en)
4. [Prathosh A.P.](https://sites.google.com/view/prathosh)


## Experiments and Models

Source code to run our experiments are present in [src](src/).

```master_script.sh``` in both [narma](src/narma/) and [pmsm](src/pmsm) folders would indicate actual runs of the experiments for 10 random seeds. This script differs slightly for these datasets as certain experiments (such as **intervention testing (noise sensitivity here)**) are exclusive to those datasets.


## Dataset

Below we present the raw time series files along with their data generating sources. The files are used to create temporal segments that are created on-the-fly while running the code.

### NARMA 

In the paper, we describe four NARMA scenarios. 

You can find the raw data to train and test (systematic generalization testing) the models in [data/narma/6](data/narma/6/).

```narma_2d_i_n0.csv``` contains the data for i<sup>th</sup> Scenario. 

```TimeSynthCombGen-spurious_corr.zip``` in [data/narma/6](data/narma/6/) contains the data generating scripts, which have been taken from TimeSynth [repository](https://github.com/TimeSynth/TimeSynth).

### Permanent Magnet Synchronous Motor (PMSM)

Raw data corresponding to PMSM described in the paper is present in [data/pmsm/](data/pmsm/). 

```GEM_PMSM.ipynb``` is the data-generating notebook which is taken from GEM [repository](https://github.com/upb-lea/gym-electric-motor)





#### Note

As this paper is a result of a lot of experiments and architectural tweaks, you might observe that the codes contain a lot more material than presented in the paper. We shall either remove those pieces with time, or add relevant explanations for such pieces in the README file. 
