import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

# Hyperparameters
from settings.global_params import global_params
from settings.hyper_params import hyperparams_fixed_tsmom, hyperparams_grid_tsmom
from settings.strategy_params import strategy_params_tsmom

# LSTM - Loss Function
@tf.keras.utils.register_keras_serializable()
def sharpe_loss(y_true, y_pred):
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
    portfolio_returns = y_pred * y_true
    mean_returns = tf.reduce_mean(portfolio_returns, axis=1)
    loss = -(mean_returns / tf.sqrt(tf.reduce_mean(tf.square(portfolio_returns - tf.reduce_mean(portfolio_returns, axis=1, keepdims=True))) + 1e-9) * tf.sqrt(252.0/strategy_params_tsmom["rebalance_freq"]))
    return tf.reduce_mean(loss)

# LSTM - Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_tsmom["LSTM"]["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# LSTM Model
class LSTMModel(tf.keras.Model):
    def __init__(self, lstm_units, dropout_rate):
        super(LSTMModel, self).__init__()
        # LSTM Layer
        self.lstm_layer = tf.keras.layers.LSTM(units=lstm_units, return_sequences=True, activation="tanh", recurrent_activation="sigmoid")
        # Dropout Layer
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        # Output Layer
        self.output_layer = tf.keras.layers.Dense(1, activation="tanh")

    def call(self, inputs, training):
        # LSTM Layer
        x = self.lstm_layer(inputs)
        # Dropout Layer
        x = self.dropout_layer(x, training=training)
        # Output Layer
        output = self.output_layer(x)
        return output

prefix = "0.46_"

class TSMOMLstm:
    def __init__(self, data_tsmom):
        # Load TSMOM data
        self.X_arr_train, self.y_arr_train, \
        self.X_arr_valid, self.y_arr_valid, \
        self.X_arr_test, self.y_arr_test, \
        self.X_indices_arr_train, self.y_indices_arr_train, \
        self.X_indices_arr_valid, self.y_indices_arr_valid, \
        self.X_indices_arr_test, self.y_indices_arr_test = data_tsmom

    def train(self, X_train, y_train, X_valid, y_valid, start_year_train, end_year_train, preload_model):
        if not preload_model:
            # Number of Assets & Batch Size
            num_assets = self.X_arr_train[0].shape[1]
            # Model Building
            class TSMOMLSTMHyperModel(kt.HyperModel):
                def build(self, hp):
                    lstm_model = LSTMModel(
                        lstm_units=hp.Choice('lstm_units', values=hyperparams_grid_tsmom["LSTM"]["units"]),
                        dropout_rate=hp.Choice('dropout_rate', values=hyperparams_grid_tsmom["LSTM"]["dropout_rate"]))
                    lstm_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=hyperparams_grid_tsmom["LSTM"]["learning_rate"])),
                        loss=sharpe_loss)
                    return lstm_model
                def fit(self, hp, model, *args, **kwargs):
                    return model.fit(
                        *args,
                        batch_size = hp.Choice('batch_size', values = hyperparams_grid_tsmom["LSTM"]["batch_size"]),
                        **kwargs,)
            # Hyperparameter Optimization
            model_tuner = kt.RandomSearch(
                TSMOMLSTMHyperModel(),
                objective = 'val_loss',
                max_trials = hyperparams_fixed_tsmom["LSTM"]["random_search_max_trials"],
                directory = 'models/hyperparameter_optimization_models',
                project_name = f'tsmom_lstm_{prefix}{start_year_train}_{end_year_train}')
            # Hyperparameter Optimization - Grid Search
            model_tuner.search(
                X_train,
                y_train,
                epochs = hyperparams_fixed_tsmom["LSTM"]["epochs"],
                validation_data = (X_valid, y_valid),
                callbacks = [early_stopping])
            # Model with Validated Hyperparameters
            model_best_params = model_tuner.get_best_hyperparameters(num_trials=1)[0]
            # Model Fit (IS)
            model = model_tuner.hypermodel.build(model_best_params)
            model.fit(
                X_train,
                y_train,
                batch_size = model_best_params.get('batch_size'),
                epochs = hyperparams_fixed_tsmom["LSTM"]["epochs"],
                validation_data = (X_valid, y_valid),
                callbacks=[early_stopping])
            # Save Model
            model.save(f'models/pretrained_models/tsmom_lstm_{prefix}{start_year_train}_{end_year_train}.tf')
        else:
            model = tf.keras.models.load_model(f'models/pretrained_models/tsmom_lstm_{prefix}{start_year_train}_{end_year_train}.tf')
        return model

    def weights(self, preload_model=False):
        num_assets = strategy_params_tsmom["num_assets"]
        # Initiate Predictions Array
        weights = []
        # Loop over Training & Prediction Intervals
        num_intervals = len(self.X_arr_train)
        for interval in range(num_intervals):
            # Start & End Year
            start_year_train = str(self.X_indices_arr_train[interval][0])[0:4]
            end_year_train = str(self.X_indices_arr_train[interval][-1])[0:4]
            # Model Training
            model = self.train(
                self.X_arr_train[interval],
                self.y_arr_train[interval], 
                self.X_arr_valid[interval], 
                self.y_arr_valid[interval],
                start_year_train,
                end_year_train,
                preload_model)
            # Model Testing
            weights_interval = model.predict(
                self.X_arr_test[interval])
            # Reshape Interval Weights
            weights_interval = weights_interval.reshape(weights_interval.shape[0], weights_interval.shape[1])
            weights_interval = np.split(weights_interval, weights_interval.shape[0] // num_assets)
            weights_interval = np.hstack(weights_interval)
            weights_interval = np.transpose(weights_interval)
            #  Stack Interval Weights
            weights = np.vstack([weights, weights_interval]) if len(weights) > 0 else weights_interval
        np.savetxt(f"data/predictions/tsmom_lstm_weights.csv", weights, delimiter=",")
        return weights