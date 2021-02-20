# MultivariateTimeSeries_Generalization

This repository is an implementation of the paper - **Systematic Generalization for Prediction Control in MultiVariate Time Series** ([arxiv](https://arxiv.org/abs/2102.05602)).

## Authors
1. [Hritik Bansal](https://sites.google.com/view/hbansal)
2. [Gantavya Bhatt](https://sites.google.com/view/gbhatt/)
3. [Pankaj Malhotra](https://scholar.google.co.in/citations?user=HP4M0MkAAAAJ&hl=en)
4. [Prathosh A.P.]()


## Dataset

### NARMA 

In the paper, we describe four NARMA scenarios. 

You can find the raw data to train and test (out-of-distribution testing) the models in [data/narma/6](data/narma/6/).

```narma_2d_i_n0.csv``` contains the data for i^{th} Scenario. 

```TimeSynthCombGen-spurious_corr.zip``` in [data/narma/6](data/narma/6/) contains the data generating scripts, which have been taken from TimeSynth [repository](https://github.com/TimeSynth/TimeSynth).

### Permanent Magnet Synchronous Motor (PMSM)

