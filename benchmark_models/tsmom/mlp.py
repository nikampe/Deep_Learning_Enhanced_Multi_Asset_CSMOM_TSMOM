import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

# Hyperparameters
from settings.global_params import global_params
from settings.hyper_params import hyperparams_fixed_tsmom, hyperparams_grid_tsmom

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_tsmom["MLP"]["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# MLP Model
class MLPModel(tf.keras.Model):
    def __init__(self, width_1, width_2, activation, dropout_rate):
        super(MLPModel, self).__init__()
        # 1st Hidden Layer
        self.first_layer = tf.keras.layers.Dense(width_1, activation=activation)
        self.first_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        # 2nd Hidden Layer
        self.second_layer = tf.keras.layers.Dense(width_2, activation=activation)
        self.second_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        # Outpout Layer
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training):
        # 1st Hidden Layer
        x = self.first_layer(inputs)
        x = self.first_dropout_layer(x, training=training)
        # 2nd Hidden Layer
        x = self.second_layer(x)
        x = self.second_dropout_layer(x, training=training)
        # Outpout Layer
        output = self.output_layer(x)
        return output

prefix = ""

class TSMOMMlp:
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
            batch_sizes = np.multiply(num_assets, hyperparams_grid_tsmom["MLP"]["mini_batch_size"]).tolist()
            # Model Building
            class TSMOMMLPHyperModel(kt.HyperModel):
                def build(self, hp):
                    mlp_tsmom = MLPModel(
                        width_1 = hp.Choice('width_1', values = hyperparams_grid_tsmom["MLP"]["width"]),
                        width_2 = hp.Choice('width_2', values = hyperparams_grid_tsmom["MLP"]["width"]),
                        activation = hp.Choice('activation', values = hyperparams_grid_tsmom["MLP"]["activation"]),
                        dropout_rate = hp.Choice('dropout_rate', values = hyperparams_grid_tsmom["MLP"]["dropout_rate"]))
                    mlp_tsmom.compile(
                        optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = hyperparams_grid_tsmom["MLP"]["learning_rate"])),
                        loss = "mse")
                    return mlp_tsmom
                def fit(self, hp, model, *args, **kwargs):
                    return model.fit(
                        *args,
                        batch_size = hp.Choice('batch_size', values = batch_sizes),
                        **kwargs,)
            # Hyperparameter Optimization
            model_tuner = kt.RandomSearch(
                TSMOMMLPHyperModel(),
                objective = 'val_loss',
                max_trials = hyperparams_fixed_tsmom["MLP"]["random_search_max_trials"],
                directory = 'models/hyperparameter_optimization_models',
                project_name = f'tsmom_mlp_{prefix}{start_year_train}_{end_year_train}')
            # Hyperparameter Optimization - Grid Search
            model_tuner.search(
                X_train,
                y_train,
                epochs = hyperparams_fixed_tsmom["MLP"]["epochs"],
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
                epochs = hyperparams_fixed_tsmom["MLP"]["epochs"],
                validation_data = (X_valid, y_valid),
                callbacks=[early_stopping])
            # Save Model
            model.save(f'models/pretrained_models/tsmom_mlp_{prefix}{start_year_train}_{end_year_train}.tf')
        else:
            model = tf.keras.models.load_model(f'models/pretrained_models/tsmom_mlp_{prefix}{start_year_train}_{end_year_train}.tf')
        return model

    def weights(self, n, preload_model=False):
        num_assets = 50
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
        return weights