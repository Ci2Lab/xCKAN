from sklearn.model_selection import train_test_split
import tomli
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def get_config(config_path):
    """
    Get configuration for the MWCD algorithm from the config.toml file

    Params
    ------
    config_path: str
        Path to the config file

    Returns
    -------
    config: dict
        Configuration dictionary
    """

    # Load config file
    with open(config_path, mode="rb") as fp:
        config = tomli.load(fp)

    # verfiy config
    if "data" not in config:
        raise ValueError("Missing data configuration in config.toml")
    else:
        if "target" not in config["data"]:
            raise ValueError("Missing target variable in data configuration in config.toml")
        if "predictors" not in config["data"]:
            raise ValueError("Missing predictors in data configuration in config.toml")
        if "split_ratio" not in config["data"]:
            raise ValueError("Missing split_ratio in data configuration in config.toml")
        if "path" not in config["data"]:
            raise ValueError("Missing path in data configuration in config.toml")
        if not os.path.exists(config["data"]["path"]):
                raise ValueError(f"File {config['data']['path']} not found. The base data folder is at ../data/. Please make sure the file exists.")
    
    if "model" not in config:
        raise ValueError("Missing model configuration in config.toml")
    else:
        if "model_types" not in config["model"]:
            raise ValueError("Missing model_types in model configuration in config.toml")

    if "misc" not in config:
        raise ValueError("Missing misc configuration in config.toml")
    else:
        if "verbose" not in config["misc"]:
            raise ValueError("Missing verbose in misc configuration in config.toml")
        if "label" not in config["misc"]:
            raise ValueError("Missing label in misc configuration in config.toml")
        if "device" not in config["misc"]:
            raise ValueError("Missing device in misc configuration in config.toml")

    return config


def load_data(config):
    """
    Load data from the path specified in the config file
    
    Params
    ------
    config: dict
        Configuration dictionary

    Returns
    -------
    X_train, X_test, y_train, y_test: pd.DataFrame
        Training and testing data splits
    """

    # Load data
    df = pd.read_csv(config["data"]["path"])
    # parse date
    df[config["data"]["date"]] = pd.to_datetime(df[config["data"]["date"]])
    df.set_index(config["data"]["date"], inplace=True)

    return df


def plot_data(df, config):
    """
    Plot the data
    
    Params
    ------
    df: pd.DataFrame
        Dataframe of predictors
    config: dict
        Configuration dictionary
    """

    # Plot data
    fig, axs = plt.subplots(2,1)
    
    axs[0].scatter(df.index, df[config["data"]["predictors"]])
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel(config["data"]["predictors"])
    axs[0].set_title("Predictors")

    axs[1].scatter(df.index, df[config["data"]["target"]])
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel(config["data"]["target"])
    axs[1].set_title("Target")

    plt.show()


def preprocess_data(X, y, split_ratio, batch_size):
    """
    Preprocess the data
    
    Params
    ------
    df: pd.DataFrame
        Dataframe of predictors

    Returns
    -------
    X_train, X_test, y_train, y_test: pd.DataFrame
        Training and testing data splits
    """

    mm = MinMaxScaler()
    ss = StandardScaler()

    X_ss = ss.fit_transform(np.array(X))
    y_mm = mm.fit_transform(np.array(y).reshape(-1, 1)) # single feature

    train_samples = int(len(X) * split_ratio)

    X_train = X_ss[:train_samples,:]
    X_test = X_ss[train_samples:,:]
    y_train = y_mm[:train_samples]
    y_test = y_mm[train_samples:]

    print("Training Shape", X_train.shape, y_train.shape)
    print("Testing Shape", X_test.shape, y_test.shape) 

    # convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # reshape input
    X_train_tensors_in = torch.reshape(X_train_tensor,   (X_train_tensor.shape[0], 1, X_train_tensor.shape[1]))
    X_test_tensors_in = torch.reshape(X_test_tensor,  (X_test_tensor.shape[0], 1, X_test_tensor.shape[1])) 

    print("Training Shape", X_train_tensors_in.shape, y_train_tensor.shape)
    print("Testing Shape", X_test_tensors_in.shape, y_test_tensor.shape) 

    x_train = torch.tensor(X_train_tensor, dtype=torch.float32)
    y_train = torch.tensor(y_train_tensor, dtype=torch.float32)
    x_test = torch.tensor(X_test_tensor, dtype=torch.float32)
    y_test = torch.tensor(y_test_tensor, dtype=torch.float32)
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataset = TensorDataset(x_test,y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return X_train_tensors_in, X_test_tensors_in, train_dataloader, test_dataloader


def plot_testing(y_test, yhat, hidden_units, batch_size, lr, n_lags, model_name, show=False, save=False):
    """
    Plot the testing results

    Args:
    -----
    y_test: np.array
        The test set target
    yhat: np.array
        The predictions
    hidden_units: int
        Number of hidden units in the dense layers
    batch_size: int
        Batch size for training
    lr: float
        Learning rate for the optimizer  
    n_lags: int
        Number of lags used for prediction  
    show: bool
        Whether to show the plot or not
    save: bool
        Whether to save the plot or not
    """
    
    plt.figure(figsize=(12, 8))
    plt.title('Time series forecasting')
    plt.plot(pd.DataFrame(y_test),  '.-', label='Measured')
    plt.plot(pd.DataFrame(yhat), '.-', label='Predictions')
    plt.legend(loc='lower right', markerscale=1)
    plt.xlabel('Date')
    plt.ylabel('Differential displacement (mm)')
    plt.grid(True)

    if save:
        plt.savefig(f"models/{model_name}/preds/filters_{hidden_units}_batch_size_{batch_size}_lr_{str(lr).split('.')[1]}_look_back_{n_lags}.png",
                    facecolor='white', edgecolor='none', bbox_inches='tight')
        
    if show:
        plt.show()

    plt.close()


def plot_training(history, hidden_units, batch_size, lr, n_lags, model_name, show=False, save=False):
    """
    Plot the training history

    Args:
    -----
    history: keras.callbacks.History
        The history of the training process
    hidden_units: int
        Number of hidden units in the dense layers
    batch_size: int
        Batch size for training
    lr: float
        Learning rate for the optimizer
    show: bool
        Whether to show the plot or not
    """

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss mae')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # save plots
    if save:
        plt.savefig(f"models/{model_name}/plots/filters_{hidden_units}_batch_size_{batch_size}_lr_{str(lr).split('.')[1]}_look_back_{n_lags}.png")
    if show:
        plt.show()

    plt.close()


def create_lagged_dataset(n_lags, df, config):
    """
    Create lagged dataset for training and testing

    Parameters:
    -----------
    n_lags: int
        Number of lags to include in the dataset
    df: pd.DataFrame
        Dataframe containing the data
    config:
        configuration file

    Returns:
    --------
    X_train: np.array
        Training predictors
    y_train: np.array
        Training target
    X_test: np.array
        Test predictors
    y_test: np.array
        Test target
    scaler: Scaler
        Set scaler to rescale predictions
    """

    # print(f'Creating lagged dataset with {n_lags} lags and {len(config["data"]["predictors"])} predictors: {config["data"]["predictors"]}')

    # split into predictors and target
    X = df[config["data"]["predictors"]]
    y = df[config["data"]["target"]].shift(-1)# predict next value based on current predictors
    y = y.dropna()

    # drop last value from predictors
    X = X.drop(X.index[-1])

    for i in range(1, n_lags):
        for col in config["data"]["predictors"]:
            X[f'{col}_lag_{i}'] = X[col].shift(i)

    X = X.dropna()
    # format y correctly by removing the valuies that were removed from X
    y = y[y.index.isin(X.index)]

    # concatenate X and y
    X['target'] = y

    values = X.values
    values = values.astype('float32')

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(values)

    scaled = scaler.transform(np.array(X))

    training_size = int(len(X)*config["data"]["split_ratio"]) - n_lags

    X_mm = np.array(scaled[:,:-1])
    y_mm = np.array(scaled[:,-1]).reshape(-1, 1)

    X_train = X_mm[:training_size,:]
    X_test = X_mm[training_size:,:]
    y_train = y_mm[:training_size]
    y_test = y_mm[training_size:]

    X_train = X_train.reshape((X_train.shape[0], n_lags, len(config["data"]["predictors"])))
    X_test = X_test.reshape((X_test.shape[0], n_lags, len(config["data"]["predictors"])))


    return X_train, y_train, X_test, y_test, scaler
