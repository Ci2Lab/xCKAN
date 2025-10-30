from turtle import width
import keras
from keras.layers import Input, Dense, LSTM, Conv1D, Bidirectional, Flatten, GRU, Dense, Dropout, Layer
import utils as ut
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import xgboost as xgb
from keras.models import Model
from datetime import datetime
from kan import *
from kan.utils import create_dataset_from_data
import warnings

def train_model(model, X_train, y_train, hidden_units, batch_size, lr, n_lags, model_name, config):
    """
    Train the model with the given hyperparameters

    Args:
    -----
    model: keras.Model
        The model to train
    hidden_units: int
        Number of hidden units in the dense layers
    batch_size: int
        Batch size for training
    lr: float
        Learning rate for the optimizer
    n_lags: int
        number of lags to take into account
    model_name: string
        name of the model for identification of the weights
    config: dict
        the configuration of the training

    Returns:
    --------
    history: keras.callbacks.History
        The history of the training process
    """


    model.compile(
        loss=tf.losses.Huber(),
        optimizer=tf.optimizers.Adam(learning_rate=lr),
        metrics=[tf.metrics.RootMeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.MeanSquaredError()],
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # what is the metric to measure
        patience=40,
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"models/{model_name}/weights/{config['misc']['result_label']}_filters_{hidden_units}_batch_size_{batch_size}_lr_{str(lr).split('.')[1]}_look_back_{n_lags}.weights.h5",
        monitor='val_loss', 
        mode='min', 
        verbose=0,
        save_best_only=True, 
        save_weights_only=True
    )
    
    history = model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=batch_size, 
        validation_split=0.1, 
        verbose=0,
        shuffle=True,
        callbacks=[model_checkpoint, early_stop],
    )
    
    return history, model

def evaluate_model(model, X_test, y_test, n_lags, predictors, scaler, print_info=False, kan_model=False):
    """
    Evaluate the model on the test set

    Args:
    -----
    model: keras.Model
        The trained model
    X_test: np.array
        The test set
    y_test: np.array
        The test set target
    n_lags: int
        Number of lags used for prediction
    predictors: list
        List of predictors
    scaler: sklearn.preprocessing.MinMaxScaler  
        The scaler used for normalization

    Returns:
    --------
    mae: float
        Mean Absolute Error
    rmse: float
        Root Mean Squared Error
    mape: float
        Mean Absolute Percentage Error
    r2: float
        R2 score
    yhat: np.array
        The predicted values
    """

    if kan_model:
        test_X_res = dataset['test_label'].detach().cpu().numpy()
        yhat = (model(dataset['test_input']).detach().cpu().numpy()).reshape((X_test.shape[0], n_lags * len(predictors)))

    else: 
        yhat = model.predict(X_test, verbose=0)
        test_X_res = X_test.reshape((X_test.shape[0], n_lags * len(predictors)))
    
    # invert scaling for forecast
    inv_yhat = np.concatenate((test_X_res,yhat), axis=1)
    # print(f'inv_yhat: {inv_yhat.shape}, yhat: {yhat.shape}, test_X_res: {test_X_res.shape}, y_test: {y_test.shape}')
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]
    # invert scaling for actual
    y_test = y_test.reshape((len(y_test), 1))
    # inv_y = np.concatenate((test_X_res, y_test), axis=1)
    # inv_y = scaler.inverse_transform(inv_y)
    # inv_y = inv_y[:, -1]

    inv_yhat = yhat
    inv_y = y_test

    # calculate MAE
    mae = mean_absolute_error(inv_y, inv_yhat)
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)
    r2 = r2_score(inv_y, inv_yhat)

    if print_info:
        print(f"Test MAE: {mae}, Test RMSE: {rmse}, Test MAPE: {mape}, Test R2: {r2}")

    return mae, rmse, mape, r2, yhat

class MLPDiamond(keras.Model):
    def __init__(self, hidden_units):
        super(MLPDiamond, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(hidden_units, activation="relu")
        self.dense2 = Dense(2*hidden_units, activation="relu")
        self.dense3 = Dense(hidden_units, activation="relu")
        self.dense4 = Dense(1)
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)
    
class MLPTriangle(keras.Model):
    def __init__(self, hidden_units):
        super(MLPTriangle, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(3*hidden_units, activation="relu")
        self.dense2 = Dense(2*hidden_units, activation="relu")
        self.dense3 = Dense(hidden_units, activation="relu")
        self.dense4 = Dense(1)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)
    
class MLPBlock(keras.Model):
    def __init__(self, hidden_units):
        super(MLPBlock, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(hidden_units, activation="relu")
        self.dense2 = Dense(hidden_units, activation="relu")
        self.dense3 = Dense(1)
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
    
class LSTM_net(keras.Model):
    def __init__(self, hidden_units, dropout=0.0):
        super(LSTM_net, self).__init__()
        self.lstm = LSTM(hidden_units, return_sequences=False, dropout=dropout)
        self.dense = Dense(1)
        
    def call(self, x):
        x = self.lstm(x)
        return self.dense(x)

class GRU_net(keras.Model):
    def __init__(self, hidden_units, dropout=0.0):
        super(GRU_net, self).__init__()
        self.gru = GRU(units=hidden_units, return_sequences=False, dropout=dropout)
        self.dense = Dense(1)
        
    def call(self, x):
        x = self.gru(x)
        return self.dense(x)    

class XLSTM(tf.keras.Model):
    def __init__(self, num_features, num_outputs, seq_length, **kwargs):
        super(XLSTM, self).__init__(**kwargs)
        self.seq_length = seq_length
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lstm_layers = [tf.keras.layers.LSTM(128, return_sequences=True) for _ in range(2)]
        self.dense = tf.keras.layers.Dense(num_outputs)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = inputs
        for lstm in self.lstm_layers:
            x = lstm(x)
            x = self.layer_norm(x)
        output = self.dense(x)
        return output[:, -1, :] 

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def Mamba(input_shape, output_dim):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(100, activation='relu', return_sequences=True))(inputs)
    x = LSTM(50, activation='relu', return_sequences=True)(x)
    x = Attention()(x)
    outputs = Dense(output_dim)(x)
    model = Model(inputs, outputs)
    return model


if __name__ == "__main__":

    # suppress pandas performance warning for iteratively appending to dataframe
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # get available devices for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")

    for config_path in [
            "config_baseline_stampa.toml",
            "config_causal_stampa.toml",
            "config_ooa_stampa.toml",
            "config_causalooa_stampa.toml"
            ]:

            # Load and verify config
            config = ut.get_config(config_path)

            # set keras backend to pytorch
            os.environ["KERAS_BACKEND"] = "torch"

            # hide tensorflow warnings
            tf.get_logger().setLevel('ERROR')
            tf.autograph.set_verbosity(0)

            # Split into training and test set
            SPLIT_RATIO = config["data"]["split_ratio"]
            START_TIME = config["data"]["start_time"]

            target = config['data']['target']
            predictors = config['data']['predictors']

            # check if result directory exists, create if not
            if not os.path.exists(f'results/{config["misc"]["label"]}/'):
                os.makedirs(f'results/{config["misc"]["label"]}/')

            # Load data
            df = pd.read_csv(config['data']['path'], index_col=config['data']['date'], parse_dates=True)

            if START_TIME == "":
                START_TIME = df.index[0]
            else:
                START_TIME = pd.to_datetime(START_TIME)
            df = df.loc[START_TIME:]

            # must not be empty for selection
            if not predictors:
                print("Causal features not set, using all features")
                predictors = df.columns
            else:
                print(f"Using causal features: {predictors}")
                # add target to predictors
                if target not in predictors:
                    predictors.append(target)

            df = df[predictors]

            training_size = int(len(df) * SPLIT_RATIO)
            n_hours = config["data"]["n_lags"]
            print(f"Training size: {training_size}")

            n = - len(predictors) + 1

            # Create and train model use tqdm to show progress bar
            from tqdm import tqdm

            all_lags = config["grid"]["all_lags"]
            all_hidden_units = config["grid"]["all_hidden_units"]
            all_lr = config["grid"]["all_lr"]
            all_batch_size = config["grid"]["all_batch_size"]
            all_estimators = config["grid"]["all_rf_estimators"]
            all_subsamples = config["grid"]["all_rf_subsamples"]
            all_sample_by_nodes = config["grid"]["all_rf_sample_by_nodes"]
            all_max_depths = config["grid"]["all_max_depths"]
            all_etas = config["grid"]["all_etas"]
            all_boost_rounds = config["grid"]["all_boost_rounds"]
            all_kan_grids = config["kan"]["all_grids"]
            all_kan_ks = config["kan"]["all_ks"]
            all_kan_lrs = config["kan"]["all_lrs"]
            all_kan_lambs = config["kan"]["all_lambs"]
            all_kan_optimisers = config["kan"]["all_optimisers"]
            all_kan_steps = config["kan"]["all_steps"]
            all_kan_widths = config["kan"]["all_widths"]
            all_models = config["model"]["model_types"] # ["MLPDiamond", "LSTM", "GRU", "XLSTM", "Mamba"] #config["model"]["model_types"]

            rf_iterations = len(all_estimators) * len(all_lr) * len(all_subsamples) * len(all_sample_by_nodes) * len(all_lags)
            xgb_iterations = len(all_max_depths) * len(all_etas) * len(all_boost_rounds) * len(all_lags)
            dl_iterations = len(all_lags) * len(all_hidden_units) * len(all_lr) * len(all_batch_size)
            kan_iterations = len(all_lags) * len(all_kan_grids) * len(all_kan_ks) * len(all_kan_lrs) * len(all_kan_lambs) * len(all_kan_optimisers) * len(all_kan_steps) * len(all_kan_widths)

            total_iterations = 0

            for model in all_models:
                if model == "RandomForest":
                    total_iterations += rf_iterations
                elif model == "XGBoost":
                    total_iterations += xgb_iterations
                elif model == "KAN":
                    total_iterations += kan_iterations
                else:
                    total_iterations += dl_iterations
            
            # loop through different hyperparameters
            with tqdm(total=total_iterations) as pbar:
                with tf.device('/gpu:0'):
                    for model in all_models:

                        print(f"Training {model} model")

                        result_file_name = f'results/{config["misc"]["label"]}/{config["misc"]["result_label"]}_{model}_results.csv'
                        print(f"Results will be saved to {result_file_name}.")

                        if model == "RandomForest":

                            for n_estimators in all_estimators:
                                for learning_rate in all_lr:
                                    for subsample in all_subsamples:
                                        for sample_by_node in all_sample_by_nodes:
                                            for n_lags in all_lags:

                                                X_train, y_train, X_test, y_test, scaler = ut.create_lagged_dataset(n_lags, df, config)
                                                
                                                dataset = {}
                                                dataset['train_input'] = X_train
                                                dataset['train_label'] = y_train
                                                dataset['test_input'] = X_test
                                                dataset['test_label'] = y_test

                                                rf_model = xgb.XGBRFRegressor(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, colsample_bynode=sample_by_node)
                                                start = datetime.now()
                                                rf_model.fit(dataset['train_input'], dataset['train_label'])
                                                end = datetime.now()

                                                # evaluate performance
                                                y_hat = rf_model.predict(dataset['test_input'])
                                                y_hat = y_hat.reshape(-1, 1)
                                                y_test = dataset['test_label'].reshape(-1, 1)

                                                # invert scaling
                                                y_hat = scaler.inverse_transform(np.concatenate(( dataset['test_input'], y_hat), axis=1))[:,-1]
                                                y_test = scaler.inverse_transform(np.concatenate(( dataset['test_input'], y_test), axis=1))[:,-1]

                                                # compute metrics
                                                mse = mean_squared_error(y_test, y_hat)
                                                mae = mean_absolute_error(y_test, y_hat)
                                                mape = mean_absolute_percentage_error(y_test, y_hat)
                                                r2 = r2_score(y_test, y_hat)
                                                rmse = np.sqrt(mse)

                                                dic = {}
                                                # Hyperparameters
                                                dic["learning_rate"] = [learning_rate]
                                                dic["n_estimators"] = [n_estimators]
                                                dic["subsample"] = [subsample]
                                                dic["look_backs"] = [n_lags]
                                                dic["colsample_bynode"] = [sample_by_node]
                                                # test_scores
                                                dic["MAE"] = [mae]
                                                dic["RMSE"] = [rmse]
                                                dic["MAPE"] = [mape]
                                                dic["R2"] = [r2]

                                                # if file does not exist, create
                                                if os.path.exists(result_file_name):
                                                    pd.DataFrame(dic).to_csv(result_file_name, mode='a', header=False, index=False)
                                                else:
                                                    pd.DataFrame(dic).to_csv(result_file_name, index=False)

                                                pbar.update(1)

                        elif model == "XGBoost":

                            for n_lags in all_lags:
                                for max_depth in all_max_depths:
                                    for eta in all_etas:
                                        for boost_rounds in all_boost_rounds:

                                            X_train, y_train, X_test, y_test, scaler = ut.create_lagged_dataset(n_lags, df, config)
                                                
                                            dataset = {}
                                            dataset['train_input'] = X_train
                                            dataset['train_label'] = y_train
                                            dataset['test_input'] = X_test
                                            dataset['test_label'] = y_test

                                            dtrain = xgb.DMatrix(dataset['train_input'], label=dataset['train_label'])
                                            dtest = xgb.DMatrix(dataset['test_input'], label=dataset['test_label'])

                                            params = {
                                                "max_depth": max_depth,
                                                "eta": eta,
                                            }

                                            start = datetime.now()                  
                                            boost_model = xgb.train(
                                                params, 
                                                dtrain, 
                                                num_boost_round=boost_rounds,    
                                            )
                                            end = datetime.now()

                                            # evaluate performance
                                            y_hat = boost_model.predict(dtest)
                                            y_hat = y_hat.reshape(-1, 1)
                                            y_test = dataset['test_label'].reshape(-1, 1)

                                            # invert scaling
                                            y_hat = scaler.inverse_transform(np.concatenate(( dataset['test_input'], y_hat), axis=1))[:,-1]
                                            y_test = scaler.inverse_transform(np.concatenate(( dataset['test_input'], y_test), axis=1))[:,-1]

                                            # compute metrics
                                            mse = mean_squared_error(y_test, y_hat)
                                            mae = mean_absolute_error(y_test, y_hat)
                                            mape = mean_absolute_percentage_error(y_test, y_hat)
                                            r2 = r2_score(y_test, y_hat)
                                            dtrain = xgb.DMatrix(dataset['train_input'], label=dataset['train_label'])
                                            dtest = xgb.DMatrix(dataset['test_input'], label=dataset['test_label'])

                                            params = {
                                                "max_depth": max_depth,
                                                "eta": eta,
                                            }

                                            start = datetime.now()                  
                                            boost_model = xgb.train(
                                                params, 
                                                dtrain, 
                                                num_boost_round=boost_rounds,    
                                            )
                                            end = datetime.now()

                                            # evaluate performance
                                            y_hat = boost_model.predict(dtest)
                                            y_hat = y_hat.reshape(-1, 1)
                                            y_test = dataset['test_label'].reshape(-1, 1)

                                            # invert scaling
                                            y_hat = scaler.inverse_transform(np.concatenate(( dataset['test_input'], y_hat), axis=1))[:,-1]
                                            y_test = scaler.inverse_transform(np.concatenate(( dataset['test_input'], y_test), axis=1))[:,-1]

                                            # compute metrics
                                            mse = mean_squared_error(y_test, y_hat)
                                            mae = mean_absolute_error(y_test, y_hat)
                                            mape = mean_absolute_percentage_error(y_test, y_hat)
                                            r2 = r2_score(y_test, y_hat)
                                            rmse = np.sqrt(mse)

                                            dic = {}

                                            # Hyperparameters
                                            dic["max_depth"] = [params['max_depth']]
                                            dic["eta"] = [params['eta']]
                                            dic["look_backs"] = [n_lags]
                                            dic["boost_rounds"] = [boost_rounds]
                                            # test_scores
                                            dic["MAE"] = [mae]
                                            dic["RMSE"] = [rmse]
                                            dic["MAPE"] = [mape]
                                            dic["R2"] = [r2]

                                            # if file does not exist, create
                                            if os.path.exists(result_file_name):
                                                pd.DataFrame(dic).to_csv(result_file_name, mode='a', header=False, index=False)
                                            else:
                                                pd.DataFrame(dic).to_csv(result_file_name, index=False)

                                            pbar.update(1)

                        elif model == "KAN":

                            for kan_grid in all_kan_grids:
                                for kan_k in all_kan_ks:
                                    for kan_lr in all_kan_lrs:
                                        for kan_lamb in all_kan_lambs:
                                            for kan_optimiser in all_kan_optimisers:
                                                for kan_step in all_kan_steps:
                                                    for kan_width in all_kan_widths:
                                                        for n_lags in all_lags:
                                                            for batch in all_batch_size:
                                                        
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
                                                                # format y correctly by removing the values that were removed from X
                                                                y = y[y.index.isin(X.index)]

                                                                training_size = int(len(X)*config["data"]["split_ratio"]) - n_lags

                                                                X_mm = np.array(X)
                                                                y_mm = np.array(y).reshape(-1, 1)

                                                                X_train = X_mm[:training_size,:]
                                                                X_test = X_mm[training_size:,:]
                                                                y_train = y_mm[:training_size]
                                                                y_test = y_mm[training_size:]

                                                                train_scaler = MinMaxScaler()
                                                                test_scaler = MinMaxScaler()
                                                                train_label_scaler = MinMaxScaler()
                                                                test_label_scaler = MinMaxScaler()
                                                                X_train = train_scaler.fit_transform(X_train)
                                                                X_test = test_scaler.fit_transform(X_test)
                                                                y_train = train_label_scaler.fit_transform(y_train)
                                                                y_test = test_label_scaler.fit_transform(y_test)

                                                                dataset = {}
                                                                dataset['train_input'] = torch.from_numpy(X_train).float()
                                                                dataset['train_label'] = torch.from_numpy(y_train).float()
                                                                dataset['test_input'] = torch.from_numpy(X_test).float()
                                                                dataset['test_label'] = torch.from_numpy(y_test).float()

                                                                # move dataset to device
                                                                dataset['train_input'] = dataset['train_input'].to(device)
                                                                dataset['train_label'] = dataset['train_label'].to(device)
                                                                dataset['test_input'] = dataset['test_input'].to(device)
                                                                dataset['test_label'] = dataset['test_label'].to(device)

                                                                net_width = [X_train.shape[1]]
                                                                if(width == 3):
                                                                    net_width.append(int(X_train.shape[1]/2))

                                                                net_width.append(1)

                                                                model = KAN(width=net_width, grid=kan_grid, k=kan_k, device=device)
                                                                results = model.fit(dataset, opt=kan_optimiser, steps=kan_step, lr=kan_lr, lamb=kan_lamb, batch=batch)

                                                                # evaluate performance
                                                                X_test_pred_inv = test_label_scaler.inverse_transform(model(dataset['test_input']).detach().cpu().numpy())
                                                                y_test_inv = test_label_scaler.inverse_transform(dataset['test_label'].detach().cpu().numpy())

                                                                r2_train = r2_score(dataset['train_label'].detach().cpu().numpy()[:, 0], model(dataset['train_input']).detach().cpu().numpy())
                                                                r2_test = r2_score(y_test_inv, X_test_pred_inv)

                                                                # RMSE
                                                                mse = mean_squared_error(y_test_inv, X_test_pred_inv)
                                                                mae = mean_absolute_error(y_test_inv, X_test_pred_inv)
                                                                mape = mean_absolute_percentage_error(y_test_inv, X_test_pred_inv)
                                                                r2 = r2_score(y_test_inv, X_test_pred_inv)
                                                                rmse = np.sqrt(mse)

                                                                dic = {}
                                                                # Hyperparameters
                                                                dic["kan_grid"] = [kan_grid]
                                                                dic["kan_k"] = [kan_k]
                                                                dic["kan_lr"] = [kan_lr]
                                                                dic["kan_lamb"] = [kan_lamb]
                                                                dic["kan_optimiser"] = [kan_optimiser]
                                                                dic["kan_step"] = [kan_step]
                                                                dic["kan_width"] = [net_width]
                                                                dic["look_backs"] = [n_lags]
                                                                dic["batch_size"] = [batch]
                                                                # test_scores
                                                                dic["MAE"] = [mae]
                                                                dic["RMSE"] = [rmse]
                                                                dic["MAPE"] = [mape]
                                                                dic["R2"] = [r2]

                                                                # if file does not exist, create
                                                                if os.path.exists(result_file_name):
                                                                    pd.DataFrame(dic).to_csv(result_file_name, mode='a', header=False, index=False)
                                                                    print(f"Saved KAN results to {result_file_name}")
                                                                else:
                                                                    pd.DataFrame(dic).to_csv(result_file_name, index=False)
                                                                    print(f"Created KAN results file at {result_file_name}")

                                                                pbar.update(1)


                        else:

                            for n_lags in all_lags:
                                X_train, y_train, X_test, y_test,scaler = ut.create_lagged_dataset(n_lags, df, config)
                                for hidden_units in all_hidden_units:
                                    for lr in all_lr:
                                        for batch_size in all_batch_size:

                                            skip = False

                                            # check if model is already trained with the parameters
                                            if os.path.exists(result_file_name):
                                                # check if one line corresponds to the current
                                                curr_res = pd.read_csv(result_file_name)
                                                params_line = f'{batch_size},{lr},{hidden_units},{n_lags}'
                                                logfile = open(result_file_name, 'r')
                                                loglist = logfile.readlines()
                                                logfile.close()
                                                found = False
                                                for line in loglist:
                                                    if params_line in line:
                                                        found = True
                                                        print(f"Model {model} with parameters {params_line} already trained.")
                                                        pbar.update(1)
                                                        skip = True
                                                        continue

                                            if skip:
                                                continue
                                        
                                            if model == 'LSTM':
                                                model_net = LSTM_net(hidden_units)
                                            elif model == 'GRU':
                                                model_net = GRU_net(hidden_units)
                                            elif model == 'MLPDiamond':
                                                model_net = MLPDiamond(hidden_units)
                                            elif model == 'MLPBlock':
                                                model_net = MLPBlock(hidden_units)
                                            elif model == 'MLPTriangle':
                                                model_net = MLPTriangle(hidden_units)
                                            elif model == 'Mamba':
                                                model_net = Mamba((n_lags, len(predictors)), 1)
                                            elif model == 'XLSTM':
                                                model_net = XLSTM(len(predictors), 1, n_lags)
                                            else:
                                                print(f"Model {model} not found")
                                                break

                                            start = pd.Timestamp.now()
                                            history, model_trained = train_model(model_net, X_train, y_train, hidden_units, batch_size, lr, n_lags, model, config)
                                            end = pd.Timestamp.now()

                                            mae, rmse, mape, r2, yhat = evaluate_model(model_trained, X_test, y_test, n_lags, predictors, scaler)
                                            print(f"Test RMSE: {rmse:.4f}, Test R2: {r2:.4f}") # Test MAE: {mae:.4f} , Test MAPE: {mape:.4f}

                                            dic = {}
                                            # Hyperparameters
                                            dic["batch_size"] = [batch_size]
                                            dic["learning_rate"] = [lr]
                                            dic["filters"] = [hidden_units]
                                            dic["look_backs"] = [n_lags]
                                            dic["training_time"] = [(end - start).total_seconds()]
                                            dic["model"] = [model]
                                            # test_scores
                                            dic["MAE"] = [mae]
                                            dic["RMSE"] = [rmse]
                                            dic["MAPE"] = [mape]
                                            dic["R2"] = [r2]

                                            # if file does not exist, create
                                            if os.path.exists(result_file_name):
                                                pd.DataFrame(dic).to_csv(result_file_name, mode='a', header=False, index=False)
                                            else:
                                                pd.DataFrame(dic).to_csv(result_file_name, index=False)

                                            pbar.update(1)
