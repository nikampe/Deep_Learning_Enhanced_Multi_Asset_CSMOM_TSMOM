import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('keras_tuner').setLevel(logging.WARNING)

# Import functions
from src.load import load_futures
from src.utils.utils import tsmom_train_valid_test_split
from src.utils.utils import returns, normalized_returns, macd_tsmom
from src.utils.utils import summary_statistics, model_performance_statistics
from src.utils.utils import plot_cumulative_returns, model_evaluation, plot_model_performances, plot_model_performances_log_scale

# Import models
from models.tsmom_encoder_decoder_transformer import TSMOMEncoderDecoderTransformer
from models.tsmom_decoder_transformer import TSMOMDecoderTransformer

# TSMOM Benchmark Models
from benchmark_models.tsmom.random import TSMOMRandom
from benchmark_models.tsmom.original_tsmom import TSMOM
from benchmark_models.tsmom.rnn import TSMOMRnn
from benchmark_models.tsmom.lstm import TSMOMLstm

# Import Global & Strategy Parameters
from settings.global_params import global_params, asset_class_map
from settings.strategy_params import strategy_params_tsmom, feature_params_tsmom

# Import Hyperparameters
from settings.hyper_params import hyperparams_fixed_tsmom, hyperparams_grid_tsmom

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

# Load TSMOM Data
def load_tsmom_data(data_raw):
    _, data_prices, data_returns = data_raw
    # TSMOM features
    data_tsmom_returns_normalized = normalized_returns(data_returns, feature_params_tsmom["normalized_periods"], global_params["vol_lookback"], global_params["vol_target"])
    data_tsmom_macd = macd_tsmom(data_prices, feature_params_tsmom)
    # Concatenation of features
    data_tsmom_features = [*data_tsmom_returns_normalized, *data_tsmom_macd]
    # TSMOM input matrix
    data_tsmom_encoder_decoder, data_tsmom_decoder = tsmom_train_valid_test_split(lookback=feature_params_tsmom["lookback"], dfs=data_tsmom_features, df_returns=data_returns, train_valid_split=global_params["train_validation_split"], rebalance_freq=strategy_params_tsmom["rebalance_freq"], vol_lookback=global_params["vol_lookback"], vol_target=global_params["vol_target"])
    return (data_tsmom_encoder_decoder, data_tsmom_decoder)

# TSMOM Encoder-Decoder Transformer - Loss Function
@tf.keras.utils.register_keras_serializable()
def sharpe_loss_encoder_decoder(y_true, y_pred):
    y_true = tf.reshape(y_true, (strategy_params_tsmom["num_assets"], int(tf.shape(y_true)[0]/strategy_params_tsmom["num_assets"])))
    y_pred = tf.reshape(y_pred, (strategy_params_tsmom["num_assets"], int(tf.shape(y_pred)[0]/strategy_params_tsmom["num_assets"])))
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
            print("\n" + "*" * 50 + f" Batch #{i+1} " + "*" * 50)
            # Interval Data
            start_year_train = str(X_indices_arr_train[i][0])[0:4]
            end_year_train = str(X_indices_arr_train[i][-1])[0:4]
            if not preload_model:
                # TSMOM Encoder-Decoder Transformer - Model Building
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
                            batch_size = hp.Choice('batch_size', values=hyperparams_grid_tsmom["Encoder_Decoder_Transformer"]["batch_size"]),
                            **kwargs,)
                # TSMOM Encoder-Decoder Transformer - Hyperparameter Optimization
                encoder_decoder_transformer_tsmom_tuner = kt.RandomSearch(
                        TSMOMEncoderDecoderTransformerHyperModel(),
                        objective = 'val_loss',
                        max_trials = hyperparams_fixed_tsmom["Encoder_Decoder_Transformer"]["random_search_max_trials"],
                        directory = 'models/hyperparameter_optimization_models',
                        project_name = f'tsmom_encoder_decoder_transformer_{start_year_train}_{end_year_train}')
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
                    epochs = 300,
                    validation_data=((X_enc_arr_valid[i], X_dec_arr_valid[i]), y_arr_valid[i]),
                    callbacks=[early_stopping_tsmom_encoder_decoder])
                encoder_decoder_transformer_tsmom_histories.append(history_tsmom)
                encoder_decoder_transformer_tsmom_labels.append(f"Transformer (Batch #{i})")
                # TSMOM Encoder-Decoder Transformer - Save Model
                encoder_decoder_transformer_tsmom.save(f'models/pretrained_models/tsmom_encoder_decoder_transformer_{start_year_train}_{end_year_train}.tf')
            else:
                # TSMOM Encoder-Decoder Transformer - Load Model
                encoder_decoder_transformer_tsmom = tf.keras.models.load_model(f'models/pretrained_models/tsmom_encoder_decoder_transformer_{start_year_train}_{end_year_train}.tf')
            # TSMOM Encoder-Decoder Transformer - Model Test
            encoder_decoder_transformer_tsmom_weights_interval = encoder_decoder_transformer_tsmom.predict(
                (X_enc_arr_test[i], X_dec_arr_test[i]),
                batch_size = 1)
            encoder_decoder_transformer_tsmom_weights_interval = np.squeeze(encoder_decoder_transformer_tsmom_weights_interval, axis=-1)
            encoder_decoder_transformer_tsmom_weights_interval = encoder_decoder_transformer_tsmom_weights_interval.reshape(int(encoder_decoder_transformer_tsmom_weights_interval.shape[0]/strategy_params_tsmom["num_assets"]),strategy_params_tsmom["num_assets"])
            # Stack Interval Weights
            encoder_decoder_transformer_tsmom_weights = np.vstack([encoder_decoder_transformer_tsmom_weights, encoder_decoder_transformer_tsmom_weights_interval]) if len(encoder_decoder_transformer_tsmom_weights) > 0 else encoder_decoder_transformer_tsmom_weights_interval
        # Save Weights in txt File
        np.savetxt(f"data/predictions/tsmom_encoder_decoder_transformer_weights.csv", encoder_decoder_transformer_tsmom_weights, delimiter=",")
    else:
        # Load Weights from txt File
        encoder_decoder_transformer_tsmom_weights = np.loadtxt(f"data/predictions/tsmom_encoder_decoder_transformer_weights.csv", delimiter=",")
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
            print("\n" + "*" * 50 + f" Batch #{i+1} " + "*" * 50)
            # Interval Data
            start_year_train = str(X_arr_indices_train[i][0])[0:4]
            end_year_train = str(X_arr_indices_train[i][-1])[0:4]
            if not preload_model:
                # TSMOM Transformer - Model Building
                num_assets = strategy_params_tsmom["num_assets"]
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
                    project_name = f'tsmom_decoder_transformer_{start_year_train}_{end_year_train}')
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
                    epochs = 300,
                    validation_data=(X_arr_valid[i], y_arr_valid[i]),
                    callbacks=[early_stopping_tsmom_decoder])
                deocder_transformer_tsmom_histories.append(decoder_transformer_tsmom_history)
                decoder_transformer_tsmom_labels.append(f"Transformer (Batch #{i})")
                # TSMOM Transformer - Save Model
                decoder_transformer_tsmom.save(f'models/pretrained_models/tsmom_decoder_transformer_{start_year_train}_{end_year_train}.tf')
            else:
                # TSMOM Transformer - Load Model
                decoder_transformer_tsmom = tf.keras.models.load_model(f'models/pretrained_models/tsmom_decoder_transformer_{start_year_train}_{end_year_train}.tf')
            # TSMOM Transformer - Model Test
            decoder_transformer_tsmom_weights_interval = decoder_transformer_tsmom.predict(
                X_arr_test[i],
                batch_size=1)
            # Reshape Interval Weights
            decoder_transformer_tsmom_weights_interval = decoder_transformer_tsmom_weights_interval.reshape(decoder_transformer_tsmom_weights_interval.shape[0], decoder_transformer_tsmom_weights_interval.shape[1])
            decoder_transformer_tsmom_weights_interval = np.split(decoder_transformer_tsmom_weights_interval, decoder_transformer_tsmom_weights_interval.shape[0] // strategy_params_tsmom["num_assets"])
            decoder_transformer_tsmom_weights_interval = np.hstack(decoder_transformer_tsmom_weights_interval)
            decoder_transformer_tsmom_weights_interval = np.transpose(decoder_transformer_tsmom_weights_interval)
            # Stack Interval Weights
            decoder_transformer_tsmom_weights = np.vstack([decoder_transformer_tsmom_weights, decoder_transformer_tsmom_weights_interval]) if len(decoder_transformer_tsmom_weights) > 0 else decoder_transformer_tsmom_weights_interval
        # Save Weights in txt File
        np.savetxt(f"data/predictions/tsmom_decoder_transformer_weights.csv", decoder_transformer_tsmom_weights, delimiter=",")
    else:
        # Load Weights from txt File
        decoder_transformer_tsmom_weights = np.loadtxt(f"data/predictions/tsmom_decoder_transformer_weights.csv", delimiter=",")
    # Print Predicted Weights
    print(f"\nTSMOM Transformer - Predicted Weights (OOS):\n", decoder_transformer_tsmom_weights)
    return decoder_transformer_tsmom_weights

def run_tsmom_benchmark_models(rebalance_days, data_returns, data_tsmom, preload_model=False):
    # Random
    weights_random_tsmom = TSMOMRandom(num_days=len(rebalance_days), num_assets=len(data_returns.columns)).weights()
    # Original TSMOM
    weights_original_3m_tsmom = TSMOM(data_returns, rebalance_days).weights(lookback=3*global_params["days_per_month"])
    weights_original_6m_tsmom = TSMOM(data_returns, rebalance_days).weights(lookback=6*global_params["days_per_month"])
    weights_original_12m_tsmom = TSMOM(data_returns, rebalance_days).weights(lookback=12*global_params["days_per_month"])
    # RNN
    weights_rnn_tsmom = TSMOMRnn(data_tsmom).weights(preload_model=preload_model)
    # LSTM
    weights_lstm_tsmom = TSMOMLstm(data_tsmom).weights(preload_model=preload_model)
    # Print results
    print(f"\nRandom (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_random_tsmom)
    print(f"\nOriginal TSMOM 3m (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_original_3m_tsmom)
    print(f"\nOriginal TSMOM 6m (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_original_6m_tsmom)
    print(f"\nOriginal TSMOM 12m (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_original_12m_tsmom)
    print(f"\nRNN (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_rnn_tsmom)
    print(f"\nLSTM (m={data_returns.shape[1]}) - Predicted Weights (OOS):\n", weights_lstm_tsmom)
    return (weights_random_tsmom, weights_original_3m_tsmom, weights_original_6m_tsmom, weights_original_12m_tsmom, weights_rnn_tsmom, weights_lstm_tsmom)

def run_tsmom_model_backtesting(investigate_data=False, preload_models=False, preload_weights=False):
    # Load data
    data_raw = load_raw_data(investigate_data=investigate_data)
    data_tsmom_encoder_decoder, data_tsmom_decoder = load_tsmom_data(data_raw)
    data_assets, data_prices, data_returns = data_raw
    _, _, _, _, _, _, _, _, _, _, X_arr_indices_test, y_arr_indices_test = data_tsmom_decoder
    # Rebalancing Days Out-of-Sample
    tsmom_rebalance_days = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in X_arr_indices_test], ignore_index=True).sort_values().reset_index(drop=True))
    tsmom_rebalance_days_evaluation = pd.DatetimeIndex(pd.concat([pd.Series(idx) for idx in y_arr_indices_test], ignore_index=True).sort_values().reset_index(drop=True))
    # Benchmark Models
    weights_benchmark = run_tsmom_benchmark_models(tsmom_rebalance_days, data_returns, data_tsmom_decoder, preload_model=preload_models)
    weights_random_tsmom, weights_original_3m_tsmom, weights_original_6m_tsmom, weights_original_12m_tsmom, weights_rnn_tsmom, weights_lstm_tsmom = weights_benchmark
    # Benchmark - Random
    scaled_returns_random_tsmom, scaled_cumulative_return_random_tsmom = model_evaluation("Random", data_returns, weights_random_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Benchmark - Original TSMOM
    scaled_returns_original_3m_tsmom, scaled_cumulative_return_original_3m_tsmom = model_evaluation("TSMOM 3M", data_returns, weights_original_3m_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    scaled_returns_original_6m_tsmom, scaled_cumulative_return_original_6m_tsmom  = model_evaluation("TSMOM 6M", data_returns, weights_original_6m_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    scaled_returns_original_12m_tsmom, scaled_cumulative_return_original_12m_tsmom  = model_evaluation("TSMOM 12M", data_returns, weights_original_12m_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Benchmark - RNN
    scaled_returns_rnn_tsmom, scaled_cumulative_return_rnn_tsmom = model_evaluation("RNN", data_returns, weights_rnn_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Benchmark - LSTM
    scaled_returns_lstm_tsmom, scaled_cumulative_return_lstm_tsmom = model_evaluation("LSTM", data_returns, weights_lstm_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Proposed TSMOM Encoder-Decoder Transformer
    weights_pred_encoder_decoder_transformer_tsmom = run_tsmom_encoder_decoder_transformer(data_tsmom_encoder_decoder, preload_model=preload_models, preload_weights=preload_weights)
    scaled_returns_encoder_decoder_transformer_tsmom, scaled_cumulative_return_encoder_decoder_transformer_tsmom = model_evaluation("Encoder-Decoder Transformer", data_returns, weights_pred_encoder_decoder_transformer_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Proposed TSMOM Decoder Transformer
    weights_pred_decoder_transformer_tsmom = run_tsmom_decoder_transformer(data_tsmom_decoder, preload_model=preload_models, preload_weights=preload_weights)
    scaled_returns_decoder_transformer_tsmom, scaled_cumulative_return_decoder_transformer_tsmom = model_evaluation("Decoder-Only Transformer", data_returns, weights_pred_decoder_transformer_tsmom, strategy_params_tsmom["rebalance_freq"], tsmom_rebalance_days, tsmom_rebalance_days_evaluation, global_params["vol_lookback"], global_params["vol_target"])
    # Plot of model performances
    plot_model_performances(
        [scaled_cumulative_return_random_tsmom,
         scaled_cumulative_return_original_3m_tsmom,
         scaled_cumulative_return_original_6m_tsmom,
         scaled_cumulative_return_original_12m_tsmom,
         scaled_cumulative_return_rnn_tsmom,
         scaled_cumulative_return_lstm_tsmom,
         scaled_cumulative_return_encoder_decoder_transformer_tsmom,
         scaled_cumulative_return_decoder_transformer_tsmom],
        [f"Random (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 3M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 6M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 12M (m={strategy_params_tsmom['num_assets']})",
         f"RNN (m={strategy_params_tsmom['num_assets']})",
         f"LSTM (m={strategy_params_tsmom['num_assets']})",
         f"Encoder-Decoder Transformer (m={strategy_params_tsmom['num_assets']})",
         f"Decoder-Only Transformer (m={strategy_params_tsmom['num_assets']})"],
        strategy="TSMOM", save=True)
    plot_model_performances_log_scale(
        [scaled_cumulative_return_random_tsmom,
         scaled_cumulative_return_original_3m_tsmom,
         scaled_cumulative_return_original_6m_tsmom,
         scaled_cumulative_return_original_12m_tsmom,
         scaled_cumulative_return_rnn_tsmom,
         scaled_cumulative_return_lstm_tsmom,
         scaled_cumulative_return_encoder_decoder_transformer_tsmom,
         scaled_cumulative_return_decoder_transformer_tsmom],
        [f"Random (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 3M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 6M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 12M (m={strategy_params_tsmom['num_assets']})",
         f"RNN (m={strategy_params_tsmom['num_assets']})",
         f"LSTM (m={strategy_params_tsmom['num_assets']})",
         f"Encoder-Decoder Transformer (m={strategy_params_tsmom['num_assets']})",
         f"Decoder-Only Transformer (m={strategy_params_tsmom['num_assets']})"],
        strategy="TSMOM", save=True)
    # Model Evaluation Statistics
    model_performance_statistics(
        [scaled_returns_random_tsmom,
         scaled_returns_original_3m_tsmom,
         scaled_returns_original_6m_tsmom,
         scaled_returns_original_12m_tsmom,
         scaled_returns_rnn_tsmom,
         scaled_returns_lstm_tsmom,
         scaled_returns_encoder_decoder_transformer_tsmom,
         scaled_returns_decoder_transformer_tsmom],
        [f"Random (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 3M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 6M (m={strategy_params_tsmom['num_assets']})",
         f"TSMOM 12M (m={strategy_params_tsmom['num_assets']})",
         f"RNN (m={strategy_params_tsmom['num_assets']})",
         f"LSTM (m={strategy_params_tsmom['num_assets']})",
         f"Encoder-Decoder Transformer (m={strategy_params_tsmom['num_assets']})",
         f"Decoder-Only Transformer (m={strategy_params_tsmom['num_assets']})"],
        strategy="TSMOM", save=True)

if __name__ == "__main__":
    run_tsmom_model_backtesting(investigate_data=False, preload_models=True, preload_weights=False)
