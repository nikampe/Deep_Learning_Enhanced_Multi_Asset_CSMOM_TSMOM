import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('keras_tuner').setLevel(logging.WARNING)

# Import functions
from src.load import load_assets, load_futures
from src.utils.utils import csmom_train_valid_test_split, tsmom_train_valid_test_split
from src.utils.utils import returns, cumulative_returns, normalized_returns, indicator_csmom, macd_csmom, macd_tsmom, asset_class
from src.utils.utils import descriptive_statistics, summary_statistics, model_performance_statistics
from src.utils.utils import plot_cumulative_returns, plot_model_learning, model_evaluation, plot_model_performances, plot_model_performances_log_scale, plot_asset_class_weights
from src.utils.utils import modify_weights, csmom_weights

# Import models
from models.csmom_transformer_ranker import CSMOMTransformerRanker
from models.csmom_transformer_reranker import CSMOMPreranker, CSMOMTransformerReranker
from models.tsmom_encoder_decoder_transformer import TSMOMEncoderDecoderTransformer
from models.tsmom_decoder_transformer import TSMOMDecoderTransformer

# CSMOM Benchmark Models
from benchmark_models.csmom.random import CSMOMRandom
from benchmark_models.csmom.original_csmom import CSMOM
from benchmark_models.csmom.macd import MACD
from benchmark_models.csmom.mlp import CSMOMMlp
from benchmark_models.csmom.listnet import ListNet
from benchmark_models.csmom.lambdamart import LambdaMART

# TSMOM Benchmark Models
from benchmark_models.tsmom.random import TSMOMRandom
from benchmark_models.tsmom.original_tsmom import TSMOM
from benchmark_models.tsmom.mlp import TSMOMMlp
from benchmark_models.tsmom.rnn import TSMOMRnn
from benchmark_models.tsmom.lstm import TSMOMLstm

# Import Global & Strategy Parameters
from settings.global_params import global_params, asset_class_map
from settings.strategy_params import strategy_params_csmom, feature_params_csmom, strategy_params_tsmom, feature_params_tsmom

# Import Hyperparameters
from settings.hyper_params import hyperparams_fixed_csmom, hyperparams_grid_csmom, hyperparams_fixed_tsmom, hyperparams_grid_tsmom

# Load Raw Data
def load_raw_data(investigate_data=False):
    # Data load
    data_assets, data_prices = load_futures(start_date=global_params["start_date"], end_date=global_params["end_date"], threshold=0.9)
    data_returns = returns(data_prices)
    # Load and concatenate CSV files
    data_returns.sort_index(axis=1, inplace=True)
    # Data investigation
    if investigate_data:
        # Summary Statistics
        summary_statistics(data_returns)
        # Raw Cumulative Returns
        plot_cumulative_returns(data_returns)
    return (data_assets, data_prices, data_returns)

# Load CSMOM DATA
def load_csmom_data(data_raw, preloaded_data):
    data_assets, data_prices, data_returns = data_raw
    # CSMOM features
    data_csmom_returns_cumulative = cumulative_returns(data_returns, feature_params_csmom["cumulative_periods"])
    data_csmom_returns_normalized = normalized_returns(data_returns, feature_params_csmom["normalized_periods"], global_params["vol_lookback"], global_params["vol_target"])
    data_csmom_macd = macd_csmom(data_prices, feature_params_csmom)
    # data_csmom_indicator = indicator_csmom(data_returns, feature_params_csmom["indicator_periods"], n=10)
    # data_csmom_asset_class = asset_class(data_returns, data_assets, asset_class_map)
    # 0.13
    # data_csmom_features = [*data_csmom_returns_cumulative, *data_csmom_returns_normalized]
    # 0.11
    # data_csmom_features = [*data_csmom_returns_cumulative, *data_csmom_returns_normalized, *data_csmom_macd]
    data_csmom_features = [*data_csmom_returns_normalized, *data_csmom_macd]
    # 0.37; 0.39; 0.63
    # data_csmom_features = [*data_csmom_returns_cumulative, *data_csmom_returns_normalized, *data_csmom_macd, *[data_csmom_macd[i].shift(period) for period in feature_params_csmom["macd_cumulative_periods"] for i in range(len(data_csmom_macd))]]
    # data_csmom_features = [*data_csmom_returns_cumulative, *data_csmom_returns_normalized, *data_csmom_macd, data_csmom_asset_class]
    # data_csmom_features = [*data_csmom_returns_cumulative, *data_csmom_returns_normalized, *data_csmom_macd, *data_csmom_indicator, data_csmom_asset_class]
    # CSMOM input matrix
    train_interval_csmom = int((global_params["days_per_month"]*global_params["months_per_year"]*strategy_params_csmom["train_interval"])/strategy_params_csmom["rebalance_freq"]) # 5y
    data_csmom = csmom_train_valid_test_split(dfs=data_csmom_features, df_returns=data_returns, train_interval=train_interval_csmom, train_valid_split=global_params["train_validation_split"], rebalance_freq=strategy_params_csmom["rebalance_freq"], vol_lookback=global_params["vol_lookback"], vol_target=global_params["vol_target"], preloaded_data=preloaded_data)
    return data_csmom

# Load TSMOM DATA
def load_tsmom_data(data_raw, preloaded_data):
    data_assets, data_prices, data_returns = data_raw
    # TSMOM features
    data_tsmom_returns_cumulative = cumulative_returns(data_returns, feature_params_tsmom["cumulative_periods"])
    data_tsmom_returns_normalized = normalized_returns(data_returns, feature_params_tsmom["normalized_periods"], global_params["vol_lookback"], global_params["vol_target"])
    data_tsmom_macd = macd_tsmom(data_prices, feature_params_tsmom)
    # data_tsmom_asset_class = asset_class(data_returns, data_assets, asset_class_map)
    # data_tsmom_features = [*data_tsmom_returns_normalized, *data_tsmom_macd]
    # 0.37 
    # data_tsmom_features = [*data_tsmom_returns_normalized]
    data_tsmom_features = [*data_tsmom_returns_cumulative, *data_tsmom_returns_normalized, *data_tsmom_macd]
    # TSMOM input matrix
    data_tsmom_encoder_decoder, data_tsmom_decoder = tsmom_train_valid_test_split(lookback=feature_params_tsmom["lookback"], dfs=data_tsmom_features, df_returns=data_returns, train_valid_split=global_params["train_validation_split"], rebalance_freq=strategy_params_tsmom["rebalance_freq"], vol_lookback=global_params["vol_lookback"], vol_target=global_params["vol_target"], preloaded_data=preloaded_data)
    return (data_tsmom_encoder_decoder, data_tsmom_decoder)

# CSMOM Transformer Ranker - Loss Function
@tf.keras.utils.register_keras_serializable()
def listnet_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_true_probs = tf.nn.softmax(y_true, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    loss = -tf.reduce_sum(y_true_probs * tf.math.log(y_pred_probs + 1e-8), axis=-1)
    return tf.reduce_mean(loss)

# CSMOM Trransformer Ranker - Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_csmom["Transformer_Ranker"]["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# CSMOM Transformer Ranker - Model Prefix
model_prefix_transformer_ranker_csmom = ""

def run_csmom_transformer_ranker(data_csmom, preload_model=False, preload_weights=False):
    # Load CSMOM data
    X_arr_train, y_arr_train, X_arr_valid, y_arr_valid, X_arr_test, y_arr_test, X_indices_arr_train, _, _, _, _, _ = data_csmom
    num_intervals = len(X_arr_train)
    if not preload_weights:
        # Model Histories
        transformer_ranker_csmom_histories, transformer_ranker_csmom_labels = [], []
        # CSMOM Ranker Weights
        transformer_ranker_csmom_weights = np.array([])
        for i in range(num_intervals):
            print("\n" + "*" * 50 + f" Batch #{i+1} " + "*" * 50)
            # Interval Data
            start_year_train = str(X_indices_arr_train[i][0])[0:4]
            end_year_train = str(X_indices_arr_train[i][-1])[0:4]
            # CSMOM Transformer Ranker - Weight Initialization
            transformer_ranker_csmom_weights_interval = np.zeros(shape=(y_arr_test[i].shape[0], y_arr_test[i].shape[1]))
            if not preload_model:
                # CSMOM Transformer Ranker - Model Building
                class CSMOMTransformerRankerHyperModel(kt.HyperModel):
                    def build(self, hp):
                        transformer_ranker_csmom = CSMOMTransformerRanker(
                            num_categories = len(asset_class_map),
                            d_model = hp.Choice('d_model', values=hyperparams_grid_csmom["Transformer_Ranker"]["d_model"]),
                            num_layers = hp.Choice('num_layers', values=hyperparams_grid_csmom["Transformer_Ranker"]["num_layers"]),
                            num_heads = hp.Choice('num_heads', values=hyperparams_grid_csmom["Transformer_Ranker"]["num_heads"]),
                            dff = hp.Choice('dff', values=hyperparams_grid_csmom["Transformer_Ranker"]["dff"]),
                            dropout_rate = hp.Choice('dropout_rate', values=hyperparams_grid_csmom["Transformer_Ranker"]["dropout_rate"]))
                        transformer_ranker_csmom.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values=hyperparams_grid_csmom["Transformer_Ranker"]["learning_rate"])),
                            loss = listnet_loss)
                        return transformer_ranker_csmom
                    def fit(self, hp, model, *args, **kwargs):
                        return model.fit(
                            *args,
                            batch_size = hp.Choice('batch_size', values=hyperparams_grid_csmom["Transformer_Ranker"]["batch_size"]),
                            **kwargs,)
                # CSMOM Transformer Ranker - Ground-Truth Decile Labels
                deciles_train = np.zeros_like(y_arr_train[i])
                for x in range(y_arr_train[i].shape[0]):
                    daily_returns = y_arr_train[i][x, :, 0]
                    decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                    daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                    deciles_train[x, :, 0] = daily_deciles
                deciles_valid = np.zeros_like(y_arr_valid[i])
                for x in range(y_arr_valid[i].shape[0]):
                    daily_returns = y_arr_valid[i][x, :, 0]
                    decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                    daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                    deciles_valid[x, :, 0] = daily_deciles
                # CSMOM Transformer Ranker - Hyperparameter Optimization
                transformer_ranker_csmom_tuner = kt.RandomSearch(
                    CSMOMTransformerRankerHyperModel(),
                    objective='val_loss',
                    max_trials=hyperparams_fixed_csmom["Transformer_Ranker"]["random_search_max_trials"],
                    directory='models/hyperparameter_optimization_models',
                    project_name=f'csmom_transformer_ranker_{model_prefix_transformer_ranker_csmom}{start_year_train}_{end_year_train}')
                # CSMOM Transformer Ranker - Hyperparameter Optimization - Grid Search
                transformer_ranker_csmom_tuner.search(
                    X_arr_train[i],
                    deciles_train,
                    epochs = hyperparams_fixed_csmom["Transformer_Ranker"]["epochs"],
                    validation_data = (X_arr_valid[i], deciles_valid),
                    callbacks = [early_stopping])
                # CSMOM Transformer Ranker - Model with Validated Hyperparameters
                transformer_ranker_csmom_best_params = transformer_ranker_csmom_tuner.get_best_hyperparameters(num_trials=1)[0]
                # CSMOM Transformer Ranker - Model Fit (IS)
                transformer_ranker_csmom = transformer_ranker_csmom_tuner.hypermodel.build(transformer_ranker_csmom_best_params)
                history_csmom = transformer_ranker_csmom.fit(
                    X_arr_train[i],
                    deciles_train,
                    batch_size = transformer_ranker_csmom_best_params.get('batch_size'),
                    epochs = hyperparams_fixed_csmom["Transformer_Ranker"]["epochs"],
                    validation_data = (X_arr_valid[i], deciles_valid),
                    callbacks=[early_stopping])
                transformer_ranker_csmom_histories.append(history_csmom)
                transformer_ranker_csmom_labels.append(f"CSMOM Transformer Ranker (Batch #{i})")
                # CSMOM Transformer Ranker - Save Model
                transformer_ranker_csmom.save(f'models/pretrained_models/csmom_transformer_ranker_{model_prefix_transformer_ranker_csmom}{start_year_train}_{end_year_train}.tf')
            else:
                transformer_ranker_csmom = tf.keras.models.load_model(f'models/pretrained_models/csmom_transformer_ranker_{model_prefix_transformer_ranker_csmom}{start_year_train}_{end_year_train}.tf')
            # CSMOM Transformer Ranker - Model Prediction (OOS)
            transfrormer_ranker_csmom_pred = transformer_ranker_csmom.predict(
                X_arr_test[i],
                batch_size = 1)
            transfrormer_ranker_csmom_pred = transfrormer_ranker_csmom_pred.reshape(transfrormer_ranker_csmom_pred.shape[0], transfrormer_ranker_csmom_pred.shape[1])
            # CSMOM Transformer Ranker - Extract Top and Bottom Assets
            top_n_assets = np.argsort(transfrormer_ranker_csmom_pred, axis=1)[:, -int(strategy_params_csmom['num_assets']/2):]
            bottom_n_assets = np.argsort(transfrormer_ranker_csmom_pred, axis=1)[:, :int(strategy_params_csmom['num_assets']/2)]
            # CSMOM Tranformer Ranker - Update Weight Predictions
            for q, top_n_assets_i in enumerate(top_n_assets):
                transformer_ranker_csmom_weights_interval[q][top_n_assets_i] = 1
            for q, bottom_n_assets_i in enumerate(bottom_n_assets):
                transformer_ranker_csmom_weights_interval[q][bottom_n_assets_i] = -1
            transformer_ranker_csmom_weights = np.vstack([transformer_ranker_csmom_weights, transformer_ranker_csmom_weights_interval]) if len(transformer_ranker_csmom_weights) > 0 else transformer_ranker_csmom_weights_interval
        # Plot Model Learnings
        plot_model_learning(model_prefix_transformer_ranker_csmom, transformer_ranker_csmom_labels, "CSMOM", "Transformer Ranker", save=True)
        # Save Weight Predictions
        np.savetxt(f"data/predictions/{model_prefix_transformer_ranker_csmom}csmom_transformer_ranker_weights.csv", transformer_ranker_csmom_weights, delimiter=",")
    else:
        # Load Weights from txt File
        transformer_ranker_csmom_weights = np.loadtxt(f"data/predictions/{model_prefix_transformer_ranker_csmom}csmom_transformer_ranker_weights.csv", delimiter=",")
    # Print Weight Predictions
    print(f"\nCSMOM Transformer Ranker - Predicted Weights (OOS):\n", transformer_ranker_csmom_weights)
    return transformer_ranker_csmom_weights

# CSMOM Transformer Re-Ranker - LTR Pre-Ranker Loss Function
@tf.keras.utils.register_keras_serializable()
def preranker_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, 50))
    y_pred = tf.reshape(y_pred, (-1, 50))
    y_true_probs = tf.nn.softmax(y_true, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    loss = -tf.reduce_sum(y_true_probs * tf.math.log(y_pred_probs + 1e-8), axis=-1)
    return tf.reduce_mean(loss)

# CSMOM Transformer Re-Ranker - Transformer Re-Ranker Loss Function
@tf.keras.utils.register_keras_serializable()
def reranker_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    y_true_probs = tf.nn.softmax(y_true, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    loss = -tf.reduce_sum(y_true_probs * tf.math.log(y_pred_probs + 1e-8), axis=-1)
    return tf.reduce_mean(loss)

# CSMOM Transformer Re-Ranker - Early Stopping Callback
early_stopping_csmom = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_csmom["Transformer_Reranker"]["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# CSMOM Transformer Re-Ranker - Model Prefix
model_prefix_transformer_reranker_csmom = ""

def run_csmom_transformer_preranker_reranker(data_csmom, preload_model=False, preload_weights=False):
    # Load CSMOM data
    X_arr_train, y_arr_train, X_arr_valid, y_arr_valid, X_arr_test, y_arr_test, X_indices_arr_train, y_indices_arr_train, X_indices_arr_valid, y_indices_arr_valid, X_indices_arr_test, y_indices_arr_test = data_csmom
    num_intervals = len(X_arr_train)
    if not preload_weights:
        # Model Histories
        listnet_preranker_csmom_histories, listnet_preranker_csmom_labels = [], []
        transformer_reranker_csmom_histories, transformer_reranker_csmom_labels = [], []
        # CSMOM Pre-Ranking Re-Ranking Model
        listnet_preranker_csmom_weights = np.array([])
        transformer_reranker_csmom_weights = np.array([])
        for i in range(num_intervals):
            if not preload_model:
                print("\n" + "*" * 50 + f" Batch #{i+1} " + "*" * 50)
                # Interval Data
                start_year_train = str(X_indices_arr_train[i][0])[0:4]
                end_year_train = str(X_indices_arr_train[i][-1])[0:4]
                # Adjusting for Last Batch (Unequal Sample Sizes)
                if X_arr_train[i].shape[0] + X_arr_valid[i].shape[0] != X_arr_test[i].shape[0]:
                    num_samples_test = X_arr_test[i].shape[0]
                    X_arr_train[i] = X_arr_train[i][:int(round(global_params["train_validation_split"]*num_samples_test,0))]
                    X_arr_valid[i] = X_arr_valid[i][:int(round((1-global_params["train_validation_split"])*num_samples_test,0))]
                    y_arr_train[i] = y_arr_train[i][:int(round(global_params["train_validation_split"]*num_samples_test,0))]
                    y_arr_valid[i] = y_arr_valid[i][:int(round((1-global_params["train_validation_split"])*num_samples_test,0))]
                # CSMOM LTR Pre-Ranker - Weight Initialization
                listnet_preranker_csmom_weights_interval = np.zeros(shape=(y_arr_test[i].shape[0], y_arr_test[i].shape[1]))
                # CSMOM LTR Pre-Ranker - Number of Assets & Batch Size
                num_assets = X_arr_train[i].shape[1]
                batch_sizes = np.multiply(num_assets, hyperparams_grid_csmom["Transformer_Reranker"]["batch_size"]).tolist()
                # CSMOM LTR Pre-Ranker - Model Building
                class CSMOMPrerankerHyperModel(kt.HyperModel):
                    def build(self, hp):
                        listnet_preranker_csmom = CSMOMPreranker(
                            width_1 = hp.Choice('width_1', values = hyperparams_grid_csmom["Transformer_Reranker"]["width"]),
                            width_2 = hp.Choice('width_2', values = hyperparams_grid_csmom["Transformer_Reranker"]["width"]),
                            activation = hp.Choice('activation', values = hyperparams_grid_csmom["Transformer_Reranker"]["activation"]),
                            dropout_rate = hp.Choice('dropout_rate', values = hyperparams_grid_csmom["Transformer_Reranker"]["dropout_rate"]))
                        listnet_preranker_csmom.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = hyperparams_grid_csmom["Transformer_Reranker"]["learning_rate"])),
                            loss = preranker_loss)
                        return listnet_preranker_csmom
                    def fit(self, hp, model, *args, **kwargs):
                        return model.fit(
                            *args,
                            batch_size = hp.Choice('batch_size', values = batch_sizes),
                            **kwargs,)
                # CSMOM LTR Pre-Ranker - Ground-Truth Decile Labels
                deciles_train = np.zeros_like(y_arr_train[i])
                for x in range(y_arr_train[i].shape[0]):
                    daily_returns = y_arr_train[i][x, :, 0]
                    decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                    daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                    deciles_train[x, :, 0] = daily_deciles
                deciles_valid = np.zeros_like(y_arr_valid[i])
                for x in range(y_arr_valid[i].shape[0]):
                    daily_returns = y_arr_valid[i][x, :, 0]
                    decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                    daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                    deciles_valid[x, :, 0] = daily_deciles
                # CSMOM LTR Pre-Ranker - Hyperparameter Optimization
                listnet_preranker_csmom_tuner = kt.RandomSearch(
                    CSMOMPrerankerHyperModel(),
                    objective = 'val_loss',
                    max_trials = hyperparams_fixed_csmom["Transformer_Reranker"]["random_search_max_trials"],
                    directory = 'models/hyperparameter_optimization_models',
                    project_name = f'csmom_listnet_preranker_{start_year_train}_{end_year_train}')
                # CSMOM LTR Pre-Ranker - Hyperparameter Optimization - Grid Search
                listnet_preranker_csmom_tuner.search(
                    X_arr_train[i].reshape(X_arr_train[i].shape[0]*X_arr_train[i].shape[1], X_arr_train[i].shape[2]),
                    deciles_train.reshape(deciles_train.shape[0]*deciles_train.shape[1], 1),
                    epochs = hyperparams_fixed_csmom["Transformer_Reranker"]["epochs"],
                    validation_data = (X_arr_valid[i].reshape(X_arr_valid[i].shape[0]*X_arr_valid[i].shape[1], X_arr_valid[i].shape[2]), deciles_valid.reshape(deciles_valid.shape[0]*deciles_valid.shape[1], 1)),
                    callbacks = [early_stopping_csmom])
                # CSMOM LTR Pre-Ranker - Model with Validated Hyperparameters
                listnet_preranker_csmom_best_params = listnet_preranker_csmom_tuner.get_best_hyperparameters(num_trials=1)[0]
                # CSMOM LTR Pre-Ranker - Model Fit (IS)
                listnet_preranker_csmom = listnet_preranker_csmom_tuner.hypermodel.build(listnet_preranker_csmom_best_params)
                listnet_preranker_csmom_history = listnet_preranker_csmom.fit(
                    X_arr_train[i].reshape(X_arr_train[i].shape[0]*X_arr_train[i].shape[1], X_arr_train[i].shape[2]),
                    deciles_train.reshape(deciles_train.shape[0]*deciles_train.shape[1], 1),
                    batch_size = listnet_preranker_csmom_best_params.get('batch_size'),
                    epochs = hyperparams_fixed_csmom["Transformer_Reranker"]["epochs"],
                    validation_data = (X_arr_valid[i].reshape(X_arr_valid[i].shape[0]*X_arr_valid[i].shape[1], X_arr_valid[i].shape[2]), deciles_valid.reshape(deciles_valid.shape[0]*deciles_valid.shape[1], 1)),
                    callbacks=[early_stopping_csmom])
                listnet_preranker_csmom_histories.append(listnet_preranker_csmom_history)
                listnet_preranker_csmom_labels.append(f"CSMOM ListNet Pre-Ranker (Batch #{i})")
                # CSMOM Transformer Pre-Ranker - Save Model
                listnet_preranker_csmom.save(f'models/pretrained_models/csmom_listnet_preranker_{model_prefix_transformer_reranker_csmom}{start_year_train}_{end_year_train}.tf')
            else:
                # CSMOM Transformer Pre-Ranker - Load Model
                listnet_preranker_csmom = tf.keras.models.load_model(f'models/pretrained_models/csmom_listnet_preranker_{model_prefix_transformer_reranker_csmom}{start_year_train}_{end_year_train}.tf')
            # CSMOM LTR Pre-Ranker - Model Prediction (OOS) (Top and Bottom Assets)
            listnet_preranker_pred = listnet_preranker_csmom.predict(
                X_arr_test[i].reshape(X_arr_test[i].shape[0]*X_arr_test[i].shape[1], X_arr_test[i].shape[2]),
                batch_size = 1)
            listnet_preranker_pred = listnet_preranker_pred.reshape(y_arr_test[i].shape[0], y_arr_test[i].shape[1])
            # CSMOM LTR Pre-Ranker - Extract Top and Bottom Assets
            top_n_assets = np.argsort(listnet_preranker_pred, axis=1)[:, -int(strategy_params_csmom['num_assets']):] # strategy_params_csmom['num_assets']/2
            bottom_n_assets = np.argsort(listnet_preranker_pred, axis=1)[:, :int(strategy_params_csmom['num_assets'])] # strategy_params_csmom['num_assets']/2
            # CSMOM LTR Pre-Ranker - Update Weight Predictions
            for q, top_n_assets_i in enumerate(top_n_assets):
                listnet_preranker_csmom_weights_interval[q][top_n_assets_i] = 1
            for q, bottom_n_assets_i in enumerate(bottom_n_assets):
                listnet_preranker_csmom_weights_interval[q][bottom_n_assets_i] = -1
            listnet_preranker_csmom_weights = np.vstack([listnet_preranker_csmom_weights, listnet_preranker_csmom_weights_interval]) if len(listnet_preranker_csmom_weights) > 0 else listnet_preranker_csmom_weights_interval
            # CSMOM Transformer Re-Ranker - Weight Initialization
            transformer_reranker_csmom_weights_interval = np.zeros(shape=(y_arr_test[i].shape[0], y_arr_test[i].shape[1]))
            # CSMOM Transformer Re-Ranker - Initialize Train, Validation, Test Data for Long & Short Portfolio
            X_train_long, y_train_long = np.empty((X_arr_train[i].shape[0], int(strategy_params_csmom['num_assets']), X_arr_train[i].shape[2])), np.empty((y_arr_train[i].shape[0], int(strategy_params_csmom['num_assets']), y_arr_train[i].shape[2])) # strategy_params_csmom['num_assets']/2
            X_valid_long, y_valid_long = np.empty((X_arr_valid[i].shape[0], int(strategy_params_csmom['num_assets']), X_arr_valid[i].shape[2])), np.empty((y_arr_valid[i].shape[0], int(strategy_params_csmom['num_assets']), y_arr_valid[i].shape[2])) # strategy_params_csmom['num_assets']/2
            X_test_long, y_test_long = np.empty((X_arr_test[i].shape[0], int(strategy_params_csmom['num_assets']), X_arr_test[i].shape[2])), np.empty((y_arr_test[i].shape[0], int(strategy_params_csmom['num_assets']), y_arr_test[i].shape[2])) # strategy_params_csmom['num_assets']/2
            X_train_short, y_train_short = np.empty((X_arr_train[i].shape[0], int(strategy_params_csmom['num_assets']), X_arr_train[i].shape[2])), np.empty((y_arr_train[i].shape[0], int(strategy_params_csmom['num_assets']), y_arr_train[i].shape[2])) # strategy_params_csmom['num_assets']/2
            X_valid_short, y_valid_short = np.empty((X_arr_valid[i].shape[0], int(strategy_params_csmom['num_assets']), X_arr_valid[i].shape[2])), np.empty((y_arr_valid[i].shape[0], int(strategy_params_csmom['num_assets']), y_arr_valid[i].shape[2])) # strategy_params_csmom['num_assets']/2
            X_test_short, y_test_short = np.empty((X_arr_test[i].shape[0], int(strategy_params_csmom['num_assets']), X_arr_test[i].shape[2])), np.empty((y_arr_test[i].shape[0], int(strategy_params_csmom['num_assets']), y_arr_test[i].shape[2]))#  strategy_params_csmom['num_assets']/2
            for j in range(X_arr_train[i].shape[0]):
                X_train_long[j], y_train_long[j] = X_arr_train[i][j, top_n_assets[j]], y_arr_train[i][j, top_n_assets[j]]
                X_train_short[j], y_train_short[j] = X_arr_train[i][j, bottom_n_assets[j]], y_arr_train[i][j, bottom_n_assets[j]]
            for j in range(X_arr_valid[i].shape[0]):
                X_valid_long[j], y_valid_long[j] = X_arr_valid[i][j, top_n_assets[j+X_arr_train[i].shape[0]]], y_arr_valid[i][j, top_n_assets[j+X_arr_train[i].shape[0]]]
                X_valid_short[j], y_valid_short[j] = X_arr_valid[i][j, bottom_n_assets[j+X_arr_train[i].shape[0]]], y_arr_valid[i][j, bottom_n_assets[j+X_arr_train[i].shape[0]]]
            for j in range(X_arr_test[i].shape[0]):
                X_test_long[j], y_test_long[j] = X_arr_test[i][j, top_n_assets[j]], y_arr_test[i][j, top_n_assets[j]]
                X_test_short[j], y_test_short[j] = X_arr_test[i][j, bottom_n_assets[j]], y_arr_test[i][j, bottom_n_assets[j]]
                transformer_reranker_csmom_weights_interval[j, top_n_assets[j]] = 1
                transformer_reranker_csmom_weights_interval[j, bottom_n_assets[j]] = -1
            # CSMOM Transformer Re-Ranker - Ground-Truth Long & Short Decile Labels
            deciles_train_long = np.zeros_like(y_train_long)
            for x in range(y_train_long.shape[0]):
                daily_returns = y_train_long[x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 50) for t in range(1, 2)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_train_long[x, :, 0] = daily_deciles
            deciles_train_short = np.zeros_like(y_train_short)
            for x in range(y_train_short.shape[0]):
                daily_returns = y_train_short[x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 50) for t in range(1, 2)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_train_short[x, :, 0] = daily_deciles
            deciles_valid_long = np.zeros_like(y_valid_long)
            for x in range(y_valid_long.shape[0]):
                daily_returns = y_valid_long[x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 50) for t in range(1, 2)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_valid_long[x, :, 0] = daily_deciles
            deciles_valid_short = np.zeros_like(y_valid_short)
            for x in range(y_valid_short.shape[0]):
                daily_returns = y_valid_short[x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 50) for t in range(1, 2)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_valid_short[x, :, 0] = daily_deciles
            # CSMOM Transformer Re-Ranker - Model Inputs
            transformer_reranker_csmom_model_input = {
                "long": {
                    "X_train": np.array(X_train_long),
                    "y_train": np.array(deciles_train_long), # y_train_long
                    "X_valid": np.array(X_valid_long),
                    "y_valid": np.array(deciles_valid_long), # y_valid_long
                    "X_test": np.array(X_test_long)
                },
                "short": {
                    "X_train": np.array(X_train_short),
                    "y_train": np.array(deciles_train_short), # y_train_short
                    "X_valid": np.array(X_valid_short),
                    "y_valid": np.array(deciles_valid_short), # y_valid_short
                    "X_test": np.array(X_test_short)}}
            for portfolio in ["long", "short"]:
                if not preload_model:
                    # CSMOM Transformer Re-Ranker - Model Building
                    class CSMOMTransformerRerankerHyperModel(kt.HyperModel):
                        def build(self, hp):
                            transformer_reranker_csmom = CSMOMTransformerReranker(
                                num_categories = len(asset_class_map),
                                d_model = hp.Choice('d_model', values=hyperparams_grid_csmom["Transformer_Reranker"]["d_model"]),
                                num_layers = hp.Choice('num_layers', values=hyperparams_grid_csmom["Transformer_Reranker"]["num_layers"]),
                                num_heads = hp.Choice('num_heads', values=hyperparams_grid_csmom["Transformer_Reranker"]["num_heads"]),
                                dff = hp.Choice('dff', values=hyperparams_grid_csmom["Transformer_Reranker"]["dff"]),
                                dropout_rate = hp.Choice('dropout_rate', values=hyperparams_grid_csmom["Transformer_Reranker"]["dropout_rate"]))
                            transformer_reranker_csmom.compile(
                                optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values=hyperparams_grid_csmom["Transformer_Reranker"]["learning_rate"])),
                                loss = reranker_loss)
                            return transformer_reranker_csmom
                        def fit(self, hp, model, *args, **kwargs):
                            return model.fit(
                                *args,
                                batch_size = hp.Choice('batch_size', values=hyperparams_grid_csmom["Transformer_Reranker"]["batch_size"]),
                                **kwargs,)
                    # CSMOM Transformer Re-Ranker - Hyperparameter Optimization
                    transformer_reranker_csmom_tuner = kt.RandomSearch(
                        # model_builder,
                        CSMOMTransformerRerankerHyperModel(),
                        objective='val_loss',
                        max_trials=hyperparams_fixed_csmom["Transformer_Reranker"]["random_search_max_trials"],
                        directory='models/hyperparameter_optimization_models',
                        project_name=f'csmom_transformer_reranker_{portfolio}_{start_year_train}_{end_year_train}')
                    # CSMOM Transformer Re-Ranker - Hyperparameter Optimization - Grid Search
                    transformer_reranker_csmom_tuner.search(
                        transformer_reranker_csmom_model_input[portfolio]["X_train"],
                        transformer_reranker_csmom_model_input[portfolio]["y_train"],
                        epochs = hyperparams_fixed_csmom["Transformer_Reranker"]["epochs"],
                        validation_data = (transformer_reranker_csmom_model_input[portfolio]["X_valid"], transformer_reranker_csmom_model_input[portfolio]["y_valid"]),
                        callbacks = [early_stopping_csmom])
                    # CSMOM Transformer Re-Ranker - Model with Validated Hyperparameters
                    transformer_reranker_csmom_best_params = transformer_reranker_csmom_tuner.get_best_hyperparameters(num_trials=1)[0]
                    # CSMOM Transformer Re-Ranker - Model Fit (IS)
                    transformer_reranker_csmom = transformer_reranker_csmom_tuner.hypermodel.build(transformer_reranker_csmom_best_params)
                    transformer_reranker_csmom_history = transformer_reranker_csmom.fit(
                        x = transformer_reranker_csmom_model_input[portfolio]["X_train"],
                        y = transformer_reranker_csmom_model_input[portfolio]["y_train"],
                        batch_size = transformer_reranker_csmom_best_params.get('batch_size'),
                        epochs = hyperparams_fixed_csmom["Transformer_Reranker"]["epochs"],
                        validation_data=(transformer_reranker_csmom_model_input[portfolio]["X_valid"], transformer_reranker_csmom_model_input[portfolio]["y_valid"]),
                        callbacks=[early_stopping_csmom])
                    transformer_reranker_csmom_histories.append(transformer_reranker_csmom_history)
                    transformer_reranker_csmom_labels.append(f"CSMOM Transformer Re-Ranker (Batch #{i}, {portfolio})")
                    # CSMOM Transformer Re-Ranker - Save Model
                    transformer_reranker_csmom.save(f'models/pretrained_models/csmom_transformer_reranker_{portfolio}_{model_prefix_transformer_reranker_csmom}{start_year_train}_{end_year_train}.tf')
                else:
                    transformer_reranker_csmom = tf.keras.models.load_model(f'models/pretrained_models/csmom_transformer_reranker_{portfolio}_{model_prefix_transformer_reranker_csmom}{start_year_train}_{end_year_train}.tf')
                # CSMOM Transformer Re-Ranker - Model Prediction (OOS)
                transformer_reranker_csmom_weights_interval_portfolio = transformer_reranker_csmom.predict(
                    transformer_reranker_csmom_model_input[portfolio]["X_test"],
                    batch_size = 1)
                transformer_reranker_csmom_weights_interval_portfolio = transformer_reranker_csmom_weights_interval_portfolio.reshape(transformer_reranker_csmom_weights_interval_portfolio.shape[0], transformer_reranker_csmom_weights_interval_portfolio.shape[1])
                for transformer_reranker_csmom_weights_interval_portfolio_i in transformer_reranker_csmom_weights_interval_portfolio:
                    largest_indices = np.argpartition(transformer_reranker_csmom_weights_interval_portfolio_i, -int(strategy_params_csmom["num_assets"]/2))[-int(strategy_params_csmom["num_assets"]/2):]
                    smallest_indices = np.argpartition(transformer_reranker_csmom_weights_interval_portfolio_i, int(strategy_params_csmom["num_assets"]/2))[:int(strategy_params_csmom["num_assets"]/2)]
                    transformer_reranker_csmom_weights_interval_portfolio_i[largest_indices] = 1 if portfolio == "long" else 0
                    transformer_reranker_csmom_weights_interval_portfolio_i[smallest_indices] = -1 if portfolio == "short" else 0
                    transformer_reranker_csmom_weights_interval_portfolio_i[np.setdiff1d(np.arange(transformer_reranker_csmom_weights_interval_portfolio_i.size), np.concatenate((largest_indices, smallest_indices)))] = 0
                # CSMOM Transformer Re-Ranker - Update Weight Predictions
                mask_indicator = 1 if portfolio == "long" else -1
                for z in range(transformer_reranker_csmom_weights_interval.shape[0]):
                    overwrite_indices = np.where(transformer_reranker_csmom_weights_interval[z] == mask_indicator)[0][:int(strategy_params_csmom['num_assets'])] # strategy_params_csmom['num_assets']/2
                    transformer_reranker_csmom_weights_interval[z, overwrite_indices] = transformer_reranker_csmom_weights_interval_portfolio[z]
            # CSMOM Transformer Re-Ranker - Stack Interval Weights
            transformer_reranker_csmom_weights = np.vstack([transformer_reranker_csmom_weights, transformer_reranker_csmom_weights_interval]) if len(transformer_reranker_csmom_weights) > 0 else transformer_reranker_csmom_weights_interval
        # Plot Model Learnings
        plot_model_learning(transformer_reranker_csmom_histories, transformer_reranker_csmom_labels, "CSMOM", "Transformer Re-Ranker", save=True)
        # Save Weight Predictions
        np.savetxt("data/predictions/csmom_transformer_reranker_weights.csv", transformer_reranker_csmom_weights, delimiter=",")
    else:
        # Load Weights from txt File
        transformer_reranker_csmom_weights = np.loadtxt(f"data/predictions/{model_prefix_transformer_ranker_csmom}csmom_transformer_ranker_weights.csv.csv", delimiter=",")
    # Print Weight Predictions
    print(f"\nCSMOM Transformer Re-Ranker - Predicted Weights (OOS):\n", transformer_reranker_csmom_weights)
    return transformer_reranker_csmom_weights

# TSMOM Encoder-Decoder Transformer - Loss Function
@tf.keras.utils.register_keras_serializable()
def sharpe_loss_encoder_decoder(y_true, y_pred):
    y_true = tf.reshape(y_true, (50, int(tf.shape(y_true)[0]/50)))
    y_pred = tf.reshape(y_pred, (50, int(tf.shape(y_pred)[0]/50)))
    portfolio_returns = y_pred * y_true
    mean_returns = tf.reduce_mean(portfolio_returns, axis=1)
    loss = -(mean_returns / tf.sqrt(tf.reduce_mean(tf.square(portfolio_returns - tf.reduce_mean(portfolio_returns, axis=1, keepdims=True))) + 1e-9) * tf.sqrt(252.0/strategy_params_tsmom["rebalance_freq"]))
    return tf.reduce_mean(loss)

# TSMOM Encoder-Decoder Transformer - Early Stopping Callback
early_stopping_tsmom_encoder_decoder = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_tsmom["Encoder_Decoder_Transformer"] ["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# TSMOM Encoder-Decoder Transformer - Model Prefix
model_prefix_encoder_decoder_tsmom_transformer = ""

def run_tsmom_encoder_decoder_transformer(data_tsmom, preload_model=False, preload_weights=False):
    # Load TSMOM data
    X_enc_arr_train, X_dec_arr_train, y_arr_train, X_enc_arr_valid, X_dec_arr_valid, y_arr_valid, X_enc_arr_test, X_dec_arr_test, y_arr_test, X_indices_arr_train, y_indices_arr_train, X_indices_arr_valid, y_indices_arr_valid, X_indices_arr_test, y_indices_arr_test = data_tsmom
    num_intervals = len(X_enc_arr_train)
    if not preload_weights:
        # Model Histories
        encoder_decoder_transformer_tsmom_histories, encoder_decoder_transformer_tsmom_labels = [], []
        # TSMOM Encoder-Decoder Transformer
        encoder_decoder_transformer_tsmom_weights = np.array([])
        for i in range(num_intervals):
            if not preload_model:
                print("\n" + "*" * 50 + f" Batch #{i+1} " + "*" * 50)
                # Interval Data
                start_year_train = str(X_indices_arr_train[i][0])[0:4]
                end_year_train = str(X_indices_arr_train[i][-1])[0:4]
                # TSMOM Encoder-Decoder Transformer - Model Building
                num_assets = 50
                batch_sizes = np.multiply(num_assets, hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["batch_size"]).tolist()
                class TSMOMEncoderDecoderTransformerHyperModel(kt.HyperModel):
                    def build(self, hp):
                        encoder_decoder_transformer_tsmom = TSMOMEncoderDecoderTransformer(
                            num_categories = len(asset_class_map),
                            d_model = hp.Choice('d_model', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["d_model"]),
                            num_layers = hp.Choice('num_layers', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["num_layers"]),
                            num_heads = hp.Choice('num_heads', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["num_heads"]),
                            dff = hp.Choice('dff', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["dff"]),
                            dropout_rate = hp.Choice('dropout_rate', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["dropout_rate"]))
                        encoder_decoder_transformer_tsmom.compile(
                            optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["learning_rate"])),
                            loss = sharpe_loss_encoder_decoder)
                        return encoder_decoder_transformer_tsmom
                    def fit(self, hp, model, *args, **kwargs):
                        return model.fit(
                            *args,
                            batch_size = hp.Choice('batch_size', values=batch_sizes),
                            **kwargs,)
                # TSMOM Encoder-Decoder Transformer - Hyperparameter Optimization
                encoder_decoder_transformer_tsmom_tuner = kt.RandomSearch(
                        TSMOMEncoderDecoderTransformerHyperModel(),
                        objective = 'val_loss',
                        max_trials = hyperparams_fixed_tsmom["Encoder_Decoder_Transformer"]["random_search_max_trials"],
                        directory = 'models/hyperparameter_optimization_models',
                        project_name = f'tsmom_encoder_decoder_transformer_{model_prefix_encoder_decoder_tsmom_transformer}{start_year_train}_{end_year_train}')
                # TSMOM Encoder-Decoder Transformer - Hyperparameter Optimization - Grid Search
                encoder_decoder_transformer_tsmom_tuner.search(
                    (X_enc_arr_train[i], X_dec_arr_train[i]),
                    y_arr_train[i],
                    epochs = hyperparams_fixed_tsmom["Encoder_Decoder_Transformer"]["epochs"],
                    validation_data=((X_enc_arr_valid[i], X_dec_arr_valid[i]), y_arr_valid[i]),
                    callbacks=[early_stopping_tsmom_encoder_decoder])
                # TSMOM Encoder-Decoder Transformer - Model with Validated Hyperparameters
                encoder_decoder_transformer_tsmom_best_params = encoder_decoder_transformer_tsmom_tuner.get_best_hyperparameters(num_trials=1)[0]
                # TSMOM Encoder-Decoder Transformer - Model Fit
                encoder_decoder_transformer_tsmom = encoder_decoder_transformer_tsmom_tuner.hypermodel.build(encoder_decoder_transformer_tsmom_best_params)
                history_tsmom = encoder_decoder_transformer_tsmom.fit(
                    x = (X_enc_arr_train[i], X_dec_arr_train[i]),
                    y = y_arr_train[i],
                    batch_size = encoder_decoder_transformer_tsmom_best_params.get('batch_size'),
                    epochs = hyperparams_fixed_tsmom["Encoder_Decoder_Transformer"]["epochs"],
                    validation_data=((X_enc_arr_valid[i], X_dec_arr_valid[i]), y_arr_valid[i]),
                    callbacks=[early_stopping_tsmom_encoder_decoder])
                encoder_decoder_transformer_tsmom_histories.append(history_tsmom)
                encoder_decoder_transformer_tsmom_labels.append(f"Transformer (Batch #{i})")
                # TSMOM Encoder-Decoder Transformer - Save Model
                encoder_decoder_transformer_tsmom.save(f'models/pretrained_models/tsmom_encoder_decoder_transformer_{model_prefix_encoder_decoder_tsmom_transformer}{start_year_train}_{end_year_train}.tf')
            else:
                # TSMOM Encoder-Decoder Transformer - Load Model
                encoder_decoder_transformer_tsmom = tf.keras.models.load_model(f'models/pretrained_models/tsmom_encoder_decoder_transformer_{model_prefix_encoder_decoder_tsmom_transformer}{start_year_train}_{end_year_train}.tf')
            # TSMOM Encoder-Decoder Transformer - Model Test
            encoder_decoder_transformer_tsmom_weights_interval = encoder_decoder_transformer_tsmom.predict(
                (X_enc_arr_test[i], X_dec_arr_test[i]),
                batch_size = 1)
            encoder_decoder_transformer_tsmom_weights_interval = np.squeeze(encoder_decoder_transformer_tsmom_weights_interval, axis=-1)
            encoder_decoder_transformer_tsmom_weights_interval = encoder_decoder_transformer_tsmom_weights_interval.reshape(int(encoder_decoder_transformer_tsmom_weights_interval.shape[0]/num_assets),num_assets)
            # Stack Interval Weights
            encoder_decoder_transformer_tsmom_weights = np.vstack([encoder_decoder_transformer_tsmom_weights, encoder_decoder_transformer_tsmom_weights_interval]) if len(encoder_decoder_transformer_tsmom_weights) > 0 else encoder_decoder_transformer_tsmom_weights_interval
        # Plot Model Learnings
        plot_model_learning(encoder_decoder_transformer_tsmom_histories, encoder_decoder_transformer_tsmom_labels, "TSMOM", "Encoder-Decoder Transformer", save=True)
        # Save Weights in txt File
        np.savetxt(f"data/predictions/{model_prefix_encoder_decoder_tsmom_transformer}tsmom_encoder_decoder_transformer_weights.csv", encoder_decoder_transformer_tsmom_weights, delimiter=",")
    else:
        # Load Weights from txt File
        encoder_decoder_transformer_tsmom_weights = np.loadtxt(f"data/predictions/{model_prefix_encoder_decoder_tsmom_transformer}tsmom_encoder_decoder_transformer_weights.csv", delimiter=",")
    # Print Predicted Weights
    print(f"\nTSMOM Encoder-Decoder Transformer - Predicted Weights (OOS):\n", encoder_decoder_transformer_tsmom_weights)
    return encoder_decoder_transformer_tsmom_weights

# TSMOM Decoder Transformer - Loss Function
@tf.keras.utils.register_keras_serializable()
def sharpe_loss_decoder(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    portfolio_returns = y_pred * y_true
    mean_returns = tf.reduce_mean(portfolio_returns, axis=1)
    loss = -(mean_returns / tf.sqrt(tf.reduce_mean(tf.square(portfolio_returns - tf.reduce_mean(portfolio_returns, axis=1, keepdims=True))) + 1e-9) * tf.sqrt(252.0/strategy_params_tsmom["rebalance_freq"]))
    return tf.reduce_mean(loss)

# TSMOM Early Stopping Callback
early_stopping_tsmom_decoder = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_tsmom["Decoder_Transformer"] ["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# TSMOM Decoder Transformer - Model Prefix
model_prefix_decoder_tsmom_transformer = ""

def run_tsmom_decoder_transformer(data_tsmom, preload_model=False, preload_weights=False):
    # Load TSMOM data
    X_arr_train, y_arr_train, X_arr_valid, y_arr_valid, X_arr_test, _, X_arr_indices_train, _, _, _, _, _ = data_tsmom
    num_intervals = len(X_arr_train)
    if not preload_weights:
        # Model Histories
        deocder_transformer_tsmom_histories, decoder_transformer_tsmom_labels = [], []
        # TSMOM Decoder Transformer - Weight Initialization
        decoder_transformer_tsmom_weights = np.array([])
        for i in range(num_intervals):
            if not preload_model:
                print("\n" + "*" * 50 + f" Batch #{i+1} " + "*" * 50)
                # Interval Data
                start_year_train = str(X_arr_indices_train[i][0])[0:4]
                end_year_train = str(X_arr_indices_train[i][-1])[0:4]
                # TSMOM Transformer - Model Building
                num_assets = 50
                batch_sizes = np.multiply(num_assets*(i+1), hyperparams_grid_tsmom["Decoder_Transformer"]["batch_size"]).tolist()
                class TSMOMDecoderTransformerHyperModel(kt.HyperModel):
                    def build(self, hp):
                        decoder_transformer_tsmom = TSMOMDecoderTransformer(
                            num_layers = hp.Choice('num_layers', values=hyperparams_grid_tsmom["Decoder_Transformer"]["num_layers"]),
                            d_model = hp.Choice('d_model', values=hyperparams_grid_tsmom["Decoder_Transformer"]["d_model"]),
                            num_heads = hp.Choice('num_heads', values=hyperparams_grid_tsmom["Decoder_Transformer"]["num_heads"]), 
                            dff = hp.Choice('dff', values=hyperparams_grid_tsmom["Decoder_Transformer"]["dff"]),
                            seq_size = X_arr_train[i].shape[1], 
                            dropout_rate = hp.Choice('dropout_rate', values=hyperparams_grid_tsmom["Decoder_Transformer"]["dropout_rate"]))
                        decoder_transformer_tsmom.compile(
                            optimizer = tf.keras.optimizers.Adam(clipvalue=1, learning_rate = hp.Choice('learning_rate', values=hyperparams_grid_tsmom["Decoder_Transformer"]["learning_rate"])),
                            loss = sharpe_loss_decoder)
                        return decoder_transformer_tsmom
                    def fit(self, hp, model, *args, **kwargs):
                        return model.fit(
                            *args,
                            batch_size = hp.Choice('batch_size', values=hyperparams_grid_tsmom["Decoder_Transformer"]["batch_size"]),
                            **kwargs,)
                # TSMOM Transformer - Hyperparameter Optimization
                decoder_transformer_tsmom_tuner = kt.RandomSearch(
                    TSMOMDecoderTransformerHyperModel(),
                    objective = 'val_loss',
                    max_trials = hyperparams_fixed_tsmom["Decoder_Transformer"]["random_search_max_trials"],
                    directory = 'models/hyperparameter_optimization_models',
                    project_name = f'tsmom_decoder_transformer_{model_prefix_decoder_tsmom_transformer}{start_year_train}_{end_year_train}')
                # TSMOM Transformer - Hyperparameter Optimization - Grid Search
                decoder_transformer_tsmom_tuner.search(
                    X_arr_train[i],
                    y_arr_train[i],
                    epochs = hyperparams_fixed_tsmom["Decoder_Transformer"]["epochs"],
                    validation_data=(X_arr_valid[i], y_arr_valid[i]),
                    callbacks=[early_stopping_tsmom_decoder])
                # TSMOM Transformer - Model with Validated Hyperparameters
                decoder_transformer_tsmom_best_params = decoder_transformer_tsmom_tuner.get_best_hyperparameters(num_trials=1)[0]
                # TSMOM Transformer - Model Fit
                decoder_transformer_tsmom = decoder_transformer_tsmom_tuner.hypermodel.build(decoder_transformer_tsmom_best_params)
                decoder_transformer_tsmom_history = decoder_transformer_tsmom.fit(
                    x = X_arr_train[i],
                    y = y_arr_train[i],
                    batch_size = decoder_transformer_tsmom_best_params.get('batch_size'),
                    epochs = hyperparams_fixed_tsmom["Decoder_Transformer"]["epochs"],
                    validation_data=(X_arr_valid[i], y_arr_valid[i]),
                    callbacks=[early_stopping_tsmom_decoder])
                deocder_transformer_tsmom_histories.append(decoder_transformer_tsmom_history)
                decoder_transformer_tsmom_labels.append(f"Transformer (Batch #{i})")
                # TSMOM Transformer - Save Model
                decoder_transformer_tsmom.save(f'models/pretrained_models/tsmom_decoder_transformer_{model_prefix_decoder_tsmom_transformer}{start_year_train}_{end_year_train}.tf')
            else:
                # TSMOM Transformer - Load Model
                decoder_transformer_tsmom = tf.keras.models.load_model(f'models/pretrained_models/tsmom_decoder_transformer_{model_prefix_decoder_tsmom_transformer}{start_year_train}_{end_year_train}.tf')
            # TSMOM Transformer - Model Test
            decoder_transformer_tsmom_weights_interval = decoder_transformer_tsmom.predict(
                X_arr_test[i],
                batch_size=1)
            # Reshape Interval Weights
            decoder_transformer_tsmom_weights_interval = decoder_transformer_tsmom_weights_interval.reshape(decoder_transformer_tsmom_weights_interval.shape[0], decoder_transformer_tsmom_weights_interval.shape[1])
            decoder_transformer_tsmom_weights_interval = np.split(decoder_transformer_tsmom_weights_interval, decoder_transformer_tsmom_weights_interval.shape[0] // num_assets)
            decoder_transformer_tsmom_weights_interval = np.hstack(decoder_transformer_tsmom_weights_interval)
            decoder_transformer_tsmom_weights_interval = np.transpose(decoder_transformer_tsmom_weights_interval)
            # decoder_transformer_tsmom_weights_interval = decoder_transformer_tsmom_weights_interval.reshape((num_assets, int(decoder_transformer_tsmom_weights_interval.shape[0]/num_assets)))
            #  Stack Interval Weights
            decoder_transformer_tsmom_weights = np.vstack([decoder_transformer_tsmom_weights, decoder_transformer_tsmom_weights_interval]) if len(decoder_transformer_tsmom_weights) > 0 else decoder_transformer_tsmom_weights_interval
        # Plot Model Learnings
        plot_model_learning(deocder_transformer_tsmom_histories, decoder_transformer_tsmom_labels, "TSMOM", "Decoder Transformer", save=True)
        # Save Weights in txt File
        np.savetxt(f"data/predictions/{model_prefix_decoder_tsmom_transformer}tsmom_decoder_transformer_weights.csv", decoder_transformer_tsmom_weights, delimiter=",")
    else:
        # Load Weights from txt File
        decoder_transformer_tsmom_weights = np.loadtxt(f"data/predictions/{model_prefix_decoder_tsmom_transformer}tsmom_decoder_transformer_weights.csv", delimiter=",")
    # Print Predicted Weights
    print(f"\nTSMOM Transformer - Predicted Weights (OOS):\n", decoder_transformer_tsmom_weights)
    return decoder_transformer_tsmom_weights

def run_csmom_benchmark_models(rebalance_days, data_returns, data_prices, data_csmom, preload_model=False):
    # Random
    weights_random_csmom = CSMOMRandom(num_days=len(rebalance_days), num_assets=len(data_returns.columns)).weights(n=strategy_params_csmom['num_assets'])
    # Original CSMO
    weights_original_3m_csmom = CSMOM(data_returns, rebalance_days).weights(lookback=3*global_params["days_per_month"], n=strategy_params_csmom['num_assets'])
    weights_original_6m_csmom = CSMOM(data_returns, rebalance_days).weights(lookback=6*global_params["days_per_month"], n=strategy_params_csmom['num_assets'])
    weights_original_12m_csmom = CSMOM(data_returns, rebalance_days).weights(lookback=12*global_params["days_per_month"], n=strategy_params_csmom['num_assets'])
    # MACD Strategy
    weights_macd_csmom = MACD(data_prices, feature_params_csmom, rebalance_days).weights(n=strategy_params_csmom['num_assets'])
    # # MLP
    # weights_mlp_csmom = CSMOMMlp(data_csmom).weights(n=strategy_params_csmom['num_assets'], preload_model=preload_model)
    # # ListNet
    # weights_ln_csmom = ListNet(data_csmom).weights(n=strategy_params_csmom['num_assets'], preload_model=preload_model)
    # # LambdaMart
    # weights_lm_csmom = LambdaMART(data_csmom).weights(n=strategy_params_csmom['num_assets'])
    # Print results
    print(f"\nRandom (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_random_csmom)
    print(f"\nOriginal CSMOM 3M (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_original_3m_csmom)
    print(f"\nOriginal CSMOM 6M (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_original_6m_csmom)
    print(f"\nOriginal CSMOM 12M (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_original_12m_csmom)
    print(f"\nMACD (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_macd_csmom)
    # print(f"\nMLP (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_mlp_csmom)
    # print(f"\nlistNet (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_ln_csmom)
    # print(f"\nLambdaMART (m={strategy_params_csmom['num_assets']}) - Predicted Weights (OOS):\n", weights_lm_csmom)
    return (weights_random_csmom, weights_original_3m_csmom, weights_original_6m_csmom, weights_original_12m_csmom, weights_macd_csmom) # , weights_mlp_csmom, weights_lm_csmom, weights_ln_csmom)

def run_tsmom_benchmark_models(rebalance_days, data_returns, data_prices, data_tsmom, preload_model=False):
    # Random
    weights_random_tsmom = TSMOMRandom(num_days=len(rebalance_days), num_assets=len(data_returns.columns)).weights()
    # Original TSMOM
    weights_original_3m_tsmom = TSMOM(data_returns, rebalance_days).weights(lookback=3*global_params["days_per_month"])
    weights_original_6m_tsmom = TSMOM(data_returns, rebalance_days).weights(lookback=6*global_params["days_per_month"])
    weights_original_12m_tsmom = TSMOM(data_returns, rebalance_days).weights(lookback=12*global_params["days_per_month"])
    # # MLP
    # weights_mlp_tsmom = TSMOMMlp(data_tsmom).weights(n=strategy_params_csmom['num_assets'], preload_model=False)
    # # RNN
    # weights_rnn_tsmom = TSMOMRnn(data_tsmom).weights(n=strategy_params_csmom['num_assets'], preload_model=False)
    # # LSTM
    # weights_lstm_tsmom = TSMOMLstm(data_tsmom).weights(n=strategy_params_csmom['num_assets'], preload_model=False)
     # Print results
    print(f"\nRandom (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_random_tsmom)
    print(f"\nOriginal TSMOM 3m (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_original_3m_tsmom)
    print(f"\nOriginal TSMOM 6m (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_original_6m_tsmom)
    print(f"\nOriginal TSMOM 12m (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_original_12m_tsmom)
    # print(f"\nMLP (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_mlp_tsmom)
    # print(f"\nRNN (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_rnn_tsmom)
    # print(f"\nLSTM (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_lstm_tsmom)
    return (weights_random_tsmom, weights_original_3m_tsmom, weights_original_6m_tsmom, weights_original_12m_tsmom) # weights_mlp_tsmom, weights_rnn_tsmom, weights_lstm_tsmom)

def run_csmom_model_backtesting(investigate_data=False, preloaded_data=False, preload_models=False, preload_weights=False):
    # Load & Unpack Data
    data_raw = load_raw_data(investigate_data=investigate_data)
    data_csmom = load_csmom_data(data_raw, preloaded_data=preloaded_data)
    _, data_prices, data_returns = data_raw
    _, _, _, _, _, _, _, _, _, _, X_indices_test, y_indices_test = data_csmom
    # Rebalancing Days OOS
    csmom_rebalance_days = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in X_indices_test], ignore_index=True).sort_values().reset_index(drop=True))
    csmom_rebalance_days_evaluation = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in y_indices_test], ignore_index=True).sort_values().reset_index(drop=True))
    # Benchmark Models
    weights_benchmark = run_csmom_benchmark_models(csmom_rebalance_days, data_returns, data_prices, data_csmom, preload_models)
    # weights_random_csmom, weights_original_3m_csmom, weights_original_6m_csmom, 0.1, 0.2, 0.3, 0.4, 0.5, weights_macd_csmom, weights_mlp_csmom, weights_lm_csmom, weights_ln_csmom = weights_benchmark
    weights_random_csmom, weights_original_3m_csmom, weights_original_6m_csmom, weights_original_12m_csmom, weights_macd_csmom = weights_benchmark
    # Benchmark Models - Random
    csmom_random_scaled_returns, csmom_random_scaled_cumulative_returns = model_evaluation("Random", data_returns, weights_random_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Benchmark Models - Original CSMOM
    csmom_original_3m_scaled_returns, csmom_original_3m_scaled_cumulative_returns = model_evaluation("CSMOM 3M", data_returns, weights_original_3m_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    csmom_original_6m_scaled_returns, csmom_original_6m_scaled_cumulative_returns  = model_evaluation("CSMOM 6M", data_returns, weights_original_6m_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    csmom_original_12m_scaled_returns, csmom_original_12m_scaled_cumulative_returns  = model_evaluation("CSMOM 12M", data_returns, weights_original_12m_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Benchmark Models - MACD
    csmom_macd_scaled_returns, csmom_macd_scaled_cumulative_returns = model_evaluation("MACD", data_returns, weights_macd_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # # Benchmark Models - MLP
    # csmom_mlp_scaled_returns, csmom_mlp_scaled_cumulative_returns = model_evaluation("MLP", data_returns, weights_mlp_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # # Benchmark Models - ListNet
    # csmom_ln_scaled_returns, csmom_ln_scaled_cumulative_returns = model_evaluation("ListNet", data_returns, weights_ln_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # # Benchmark Models - LambdaMART
    # csmom_lm_scaled_returns, csmom_lm_scaled_cumulative_returns = model_evaluation("LambdaMART", data_returns, weights_lm_csmom, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Proposed CSMOM Transformer Ranker
    csmom_transformer_ranker_weights = run_csmom_transformer_ranker(data_csmom, preload_model=preload_models, preload_weights=preload_weights)
    csmom_transformer_ranker_scaled_returns, csmom_transformer_ranker_scaled_cumulative_returns = model_evaluation("Transformer Ranker", data_returns, csmom_transformer_ranker_weights, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Proposed CSMOM Transformer Re-Ranker
    csmom_transformer_reranker_weights = run_csmom_transformer_preranker_reranker(data_csmom, preload_model=preload_models, preload_weights=preload_weights)
    csmom_transformer_reranker_scaled_returns, csmom_transformer_reranker_scaled_cumulative_returns = model_evaluation("Transformer Ranker", data_returns, csmom_transformer_reranker_weights, strategy_params_csmom["rebalance_freq"], csmom_rebalance_days, csmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Plot of Model Performances (Linear Scale)
    plot_model_performances(
        [csmom_random_scaled_cumulative_returns,
         csmom_original_3m_scaled_cumulative_returns,
         csmom_original_6m_scaled_cumulative_returns,
         csmom_original_12m_scaled_cumulative_returns,
         csmom_macd_scaled_cumulative_returns,
        #  csmom_mlp_scaled_cumulative_returns,
        #  csmom_ln_scaled_cumulative_returns,
        #  csmom_lm_scaled_cumulative_returns,
         csmom_transformer_ranker_scaled_cumulative_returns,
         csmom_transformer_reranker_scaled_cumulative_returns],
        [f"Random (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 3M (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 6M (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 12M (m={strategy_params_csmom['num_assets']})",
         f"MACD (m={strategy_params_csmom['num_assets']})",
        #  f"MLP (m={strategy_params_csmom['num_assets']})",
        #  f"ListNet (m={strategy_params_csmom['num_assets']})",
        #  f"LambdaMART (m={strategy_params_csmom['num_assets']})",
         f"Transformer Ranker (m={strategy_params_csmom['num_assets']})",
         f"Transformer Re-Ranker (m={strategy_params_csmom['num_assets']})"],
        strategy="CSMOM", save=True)
    # Plot of Model Performances (Logarithmic Scale)
    plot_model_performances_log_scale(
        [csmom_random_scaled_cumulative_returns,
         csmom_original_3m_scaled_cumulative_returns,
         csmom_original_6m_scaled_cumulative_returns,
         csmom_original_12m_scaled_cumulative_returns,
         csmom_macd_scaled_cumulative_returns,
        #  csmom_mlp_scaled_cumulative_returns,
        #  csmom_ln_scaled_cumulative_returns,
        #  csmom_lm_scaled_cumulative_returns,
         csmom_transformer_ranker_scaled_cumulative_returns,
         csmom_transformer_reranker_scaled_cumulative_returns],
        [f"Random (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 3M (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 6M (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 12M (m={strategy_params_csmom['num_assets']})",
         f"MACD (m={strategy_params_csmom['num_assets']})",
        #  f"MLP (m={strategy_params_csmom['num_assets']})",
        #  f"ListNet (m={strategy_params_csmom['num_assets']})",
        #  f"LambdaMART (m={strategy_params_csmom['num_assets']})",
         f"Transformer Ranker (m={strategy_params_csmom['num_assets']})",
         f"Transformer Re-Ranker (m={strategy_params_csmom['num_assets']})"],
        strategy="CSMOM", save=True)
    # Model Evaluation Statistics
    model_performance_statistics(
        [csmom_random_scaled_returns,
         csmom_original_3m_scaled_returns,
         csmom_original_6m_scaled_returns,
         csmom_original_12m_scaled_returns,
         csmom_macd_scaled_returns,
        #  csmom_mlp_scaled_returns,
        #  csmom_ln_scaled_returns,
        #  csmom_lm_scaled_returns,
         csmom_transformer_ranker_scaled_returns,
         csmom_transformer_reranker_scaled_returns],
        [f"Random (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 3M (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 6M (m={strategy_params_csmom['num_assets']})",
         f"CSMOM 12M (m={strategy_params_csmom['num_assets']})",
         f"MACD (m={strategy_params_csmom['num_assets']})",
        #  f"MLP (m={strategy_params_csmom['num_assets']})",
        #  f"ListNet (m={strategy_params_csmom['num_assets']})",
        #  f"LambdaMART (m={strategy_params_csmom['num_assets']})",
         f"Transformer Ranker (m={strategy_params_csmom['num_assets']})",
         f"Transformer Re-Ranker (m={strategy_params_csmom['num_assets']})"],
        strategy="CSMOM", save=True)

def run_tsmom_model_backtesting(investigate_data=False, preloaded_data=False, preload_models=False, preload_weights=False):
    # Load data
    data_raw = load_raw_data(investigate_data=investigate_data)
    data_tsmom_encoder_decoder, data_tsmom_decoder = load_tsmom_data(data_raw, preloaded_data=preloaded_data)
    data_assets, data_prices, data_returns = data_raw
    _, _, _, _, _, _, _, _, _, _, X_arr_indices_test, y_arr_indices_test = data_tsmom_decoder
    # Rebalancing Days Out-of-Sample
    tsmom_rebalance_days = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in X_arr_indices_test], ignore_index=True).sort_values().reset_index(drop=True))
    tsmom_rebalance_days_evaluation = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in y_arr_indices_test], ignore_index=True).sort_values().reset_index(drop=True))
    # Benchmark Models
    weights_benchmark = run_tsmom_benchmark_models(tsmom_rebalance_days, data_returns, data_prices, data_tsmom_decoder, preload_model=preload_models)
    # weights_random_tsmom, weights_original_3m_tsmom, weights_original_6m_tsmom, weights_original_12m_tsmom, weights_mlp_tsmom, weights_rnn_tsmom, weights_lstm_tsmom = weights_benchmark
    weights_random_tsmom, weights_original_3m_tsmom, weights_original_6m_tsmom, weights_original_12m_tsmom = weights_benchmark
    # Benchmark - Random
    scaled_returns_random_tsmom, scaled_cumulative_return_random_tsmom = model_evaluation("Random", data_returns, weights_random_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Benchmark - Original TSMOM
    scaled_returns_original_3m_tsmom, scaled_cumulative_return_original_3m_tsmom = model_evaluation("TSMOM 3M", data_returns, weights_original_3m_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    scaled_returns_original_6m_tsmom, scaled_cumulative_return_original_6m_tsmom  = model_evaluation("TSMOM 6M", data_returns, weights_original_6m_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    scaled_returns_original_12m_tsmom, scaled_cumulative_return_original_12m_tsmom  = model_evaluation("TSMOM 12M", data_returns, weights_original_12m_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # # Benchmark - MLP
    # scaled_returns_mlp_tsmom, scaled_cumulative_return_mlp_tsmom = model_evaluation("Random", data_returns, weights_mlp_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # # Benchmark - RNN
    # scaled_returns_rnn_tsmom, scaled_cumulative_return_rnn_tsmom = model_evaluation("Random", data_returns, weights_mlp_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # # Benchmark - LSTM
    # scaled_returns_lstm_tsmom, scaled_cumulative_return_lstm_tsmom = model_evaluation("Random", data_returns, weights_mlp_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Proposed TSMOM Encoder-Decoder Transformer
    weights_pred_encoder_decoder_transformer_tsmom = run_tsmom_encoder_decoder_transformer(data_tsmom_encoder_decoder, preload_model=preload_models, preload_weights=preload_weights)
    scaled_returns_encoder_decoder_transformer_tsmom, scaled_cumulative_return_encoder_decoder_transformer_tsmom = model_evaluation("Transformer", data_returns, weights_pred_encoder_decoder_transformer_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Proposed TSMOM Decoder Transformer
    weights_pred_decoder_transformer_tsmom = run_tsmom_decoder_transformer(data_tsmom_decoder, preload_model=preload_models, preload_weights=preload_weights)
    scaled_returns_decoder_transformer_tsmom, scaled_cumulative_return_decoder_transformer_tsmom = model_evaluation("Transformer", data_returns, weights_pred_decoder_transformer_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Plot of model performances
    plot_model_performances(
        [scaled_cumulative_return_random_tsmom,
         scaled_cumulative_return_original_3m_tsmom,
         scaled_cumulative_return_original_6m_tsmom,
         scaled_cumulative_return_original_12m_tsmom,
        #  scaled_cumulative_return_mlp_tsmom,
        #  scaled_cumulative_return_rnn_tsmom,
        #  scaled_cumulative_return_lstm_tsmom,
         scaled_cumulative_return_encoder_decoder_transformer_tsmom,
         scaled_cumulative_return_decoder_transformer_tsmom],
        [f"Random (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 3M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 6M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 12M (m={strategy_params_tsmom['num_assets']})",
        #  f"MLP (m={strategy_params_tsmom['num_assets']})",
        #  f"RNN (m={strategy_params_tsmom['num_assets']})",
        #  f"LSTM (m={strategy_params_tsmom['num_assets']})",
         f"Encoder-Decoder Transformer (m={strategy_params_tsmom['num_assets']})",
         f"Decoder-Only Transformer (m={strategy_params_tsmom['num_assets']})"],
        strategy="TSMOM", save=True)
    plot_model_performances_log_scale(
        [scaled_cumulative_return_random_tsmom,
         scaled_cumulative_return_original_3m_tsmom,
         scaled_cumulative_return_original_6m_tsmom,
         scaled_cumulative_return_original_12m_tsmom,
        #  scaled_cumulative_return_mlp_tsmom,
        #  scaled_cumulative_return_rnn_tsmom,
        #  scaled_cumulative_return_lstm_tsmom,
         scaled_cumulative_return_encoder_decoder_transformer_tsmom,
         scaled_cumulative_return_decoder_transformer_tsmom],
        [f"Random (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 3M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 6M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 12M (m={strategy_params_tsmom['num_assets']})",
        #  f"MLP (m={strategy_params_tsmom['num_assets']})",
        #  f"RNN (m={strategy_params_tsmom['num_assets']})",
        #  f"LSTM (m={strategy_params_tsmom['num_assets']})",
         f"Encoder-Decoder Transformer (m={strategy_params_tsmom['num_assets']})",
         f"Decoder-Only Transformer (m={strategy_params_tsmom['num_assets']})"],
        strategy="TSMOM", save=True)
    # Model Evaluation Statistics
    model_performance_statistics(
        [scaled_returns_random_tsmom,
         scaled_returns_original_3m_tsmom,
         scaled_returns_original_6m_tsmom,
         scaled_returns_original_12m_tsmom,
        #  scaled_returns_mlp_tsmom,
        #  scaled_returns_rnn_tsmom,
        #  scaled_returns_lstm_tsmom,
         scaled_returns_encoder_decoder_transformer_tsmom,
         scaled_returns_decoder_transformer_tsmom],
        [f"Random (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 3M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 6M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 12M (m={strategy_params_tsmom['num_assets']})",
        #  f"MLP (m={strategy_params_tsmom['num_assets']})",
        #  f"RNN (m={strategy_params_tsmom['num_assets']})",
        #  f"LSTM (m={strategy_params_tsmom['num_assets']})",
         f"Encoder-Decoder Transformer (m={strategy_params_tsmom['num_assets']})",
         f"Decoder-Only Transformer (m={strategy_params_tsmom['num_assets']})"],
        strategy="TSMOM", save=True)

if __name__ == "__main__":
    # CSMOM
    run_csmom_model_backtesting(investigate_data=False, preloaded_data=False, preload_models=False, preload_weights=False)
    # TSMOM
    # run_tsmom_model_backtesting(investigate_data=False, preloaded_data=False, preload_models=False, preload_weights=False)


