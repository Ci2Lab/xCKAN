# xCKAN

Improving Surface Displacement Prediction using Explainable AI and Causal Feature Selection

![xCKAN Overview](graphical_abstract.png)


## Table of Contents

[1. Getting Started](#get-started)

- [1.1. Required Packages](#required-packages)
- [1.2. Example](#example)

[2. Options](#2-options)
- [model](#model)
- [grid](#grid)
- [kan](#kan)
- [data](#data)
- [misc](#misc)


## 1. Get Started

In order to run the example, first install the [required packages](#required-packages), then run the [Example](#example) using the provided example dataset.
Causal Discovery and splitting the Last Segment is based on the [Multi-Window Causal Discovery](https://github.com/Ci2Lab/MWCD).


### 1.1. Required Packages

To install the required Python packages, run `pip install -r requirements.txt` to install the packages specified in the `requirements.txt` file. Furthermore, the `gcastle` package is set to use PyTorch as backend. Install PyTorch [using the PyTorch Get Started instructions](https://pytorch.org/get-started/locally/).


### 1.2. Example

To run the example, execute `python main.py`. The main script will load the example dataset from `/data`. To change the options the framework uses, change any of the options specified under [Options](#options).


## 2. Options

The xCKAN offers a variety of options to adapt it to your dataset. Mainly, you can specify the parameter space for the grid search.

The base setup uses four different configuration files for the four different data configurations used in the ablation study: (1) **baseline** for full time range and all input features, (2) **causal** for full time range with features reduced to causal features only, (3) **ooa** short for Onset Of Accelerations where only samples in the Last Segment of the time series are used from all features and lastly (4) **causalooa** where in the Last Segment only samples from causal features are being used to train the model.

The configuration files use the [TOML format](https://github.com/toml-lang/toml) to describe the configuration for the xCKAN framework runs.

Per configuration file, 5 sections are specified:

### `model`
specifies the models to be grid searched and compared for prediction performance. Uses the sub-options:
- `model_types`
List of models to be hyperparameter tuned. Available models
    - MLPDiamond
    - MLPTriangle
    - MLPBlock
    - LSTM
    - GRU
    - XGBoost
    - RandomForest
    - KAN

### `grid`
Grids for the grid search
- `all_rf_estimators`: number of trees in the forest. For `RandomForest` only.
- `all_rf_subsamples`: Subsample ratio of the training instances.
- `all_rf_sample_by_nodes`: subsample ratio of columns for each node
- `all_max_depths`: Maximum depth of a tree.
- `all_etas`: Step size shrinkage for boosting steps. For `XGBoost` only.
- `all_boost_rounds`: boost rounds
- `all_lags`: time lags to be considered
- `all_lr`: learning ratessubsample ratio of columns for each node
- `all_batch_size`: batch sizes
- `all_optimisers`: optimisers
- `all_hidden_units`: neurons on hidden layers

### `kan`
Grids for KAN optimisation only. Fields used in this option:
- `all_grids`: number of grid points
- `all_ks`: spline orders
- `all_lrs`: learning rates
- `all_lambs`: reguralization
- `all_optimisers`: optimisers, tested: LBFGS, Adam, SGD
- `all_steps`: epochs for training
- `all_widths`: network widths, currently midlayers are automatically populated as being half the size of the input. Can be changed in the `main.py` file.


### `data`
Description of the dataset.
The following options are used as sub-options:

- `path`: path to the dataset. By default the data is located in the directory `"./data/stampa.csv"`. Note that the path is relative to the location of the configuration file.
- `target`: target variable name. This has to be a column that is present in the specified dataset file.
- `predictors`: list of variables used to predict the `target` variable. Importantly this should only contain causal variables for the causal configurations.
- `date`: name of the column indicating the sample timestamps.
- `split_ratio`: train/test split ratio.
- `start_time`: Beginning of the training data set. This is used to indicate the start of the Last Segment for `ooa` and `causalooa` configurations. If left empty, all samples will be used (i.e. for `baseline` and `causal` configurations).

### `misc`
This section collects options used to manage the data, inputs and outputs.
- `verbose`: controls the level of verbosity for the output streamed to stdout.
- `label`: data label used for the result files
- `result_label`: data label used to name result files
- `device`: may be used to specify running on a specific device.

