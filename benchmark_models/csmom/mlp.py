import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

# Hyperparameters
from settings.global_params import global_params
from settings.hyper_params import hyperparams_fixed_csmom, hyperparams_grid_csmom

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_csmom["MLP"]["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# ListNet Model
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

class CSMOMMlp:
    def __init__(self, data_csmom):
        # Load CSMOM data
        self.X_arr_train, self.y_arr_train, \
        self.X_arr_valid, self.y_arr_valid, \
        self.X_arr_test, self.y_arr_test, \
        self.X_indices_arr_train, self.y_indices_arr_train, \
        self.X_indices_arr_valid, self.y_indices_arr_valid, \
        self.X_indices_arr_test, self.y_indices_arr_test = data_csmom

    def train(self, X_train, y_train, X_valid, y_valid, start_year_train, end_year_train, preload_model):
        if not preload_model:
            # Number of Assets & Batch Size
            num_assets = self.X_arr_train[0].shape[1]
            batch_sizes = np.multiply(num_assets, hyperparams_grid_csmom["MLP"]["mini_batch_size"]).tolist()
            # Model Building
            class CSMOMMLPHyperModel(kt.HyperModel):
                def build(self, hp):
                    preranker_csmom = MLPModel(
                        width_1 = hp.Choice('width_1', values = hyperparams_grid_csmom["MLP"]["width"]),
                        width_2 = hp.Choice('width_2', values = hyperparams_grid_csmom["MLP"]["width"]),
                        activation = hp.Choice('activation', values = hyperparams_grid_csmom["MLP"]["activation"]),
                        dropout_rate = hp.Choice('dropout_rate', values = hyperparams_grid_csmom["MLP"]["dropout_rate"]))
                    preranker_csmom.compile(
                        optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = hyperparams_grid_csmom["MLP"]["learning_rate"])),
                        loss = "mse")
                    return preranker_csmom
                def fit(self, hp, model, *args, **kwargs):
                    return model.fit(
                        *args,
                        batch_size = hp.Choice('batch_size', values = batch_sizes),
                        **kwargs,)
            # Hyperparameter Optimization
            model_tuner = kt.RandomSearch(
                CSMOMMLPHyperModel(),
                objective = 'val_loss',
                max_trials = hyperparams_fixed_csmom["MLP"]["random_search_max_trials"],
                directory = 'models/hyperparameter_optimization_models',
                project_name = f'{prefix}csmom_mlp_{start_year_train}_{end_year_train}')
            # Hyperparameter Optimization - Grid Search
            model_tuner.search(
                X_train,
                y_train,
                epochs = hyperparams_fixed_csmom["MLP"]["epochs"],
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
                epochs = hyperparams_fixed_csmom["MLP"]["epochs"],
                validation_data = (X_valid, y_valid),
                callbacks=[early_stopping])
            # Save Model
            model.save(f'models/pretrained_models/{prefix}csmom_mlp_{start_year_train}_{end_year_train}.tf')
        else:
            model = tf.keras.models.load_model(f'models/pretrained_models/{prefix}csmom_mlp_{start_year_train}_{end_year_train}.tf')
        return model

    def weights(self, n, preload_model=False):
        # Initiate Predictions Array
        weights = []
        # Loop over Training & Prediction Intervals
        num_intervals = len(self.X_arr_train)
        for interval in range(num_intervals):
            # Start & End Year
            start_year_train = str(self.X_indices_arr_train[interval][0])[0:4]
            end_year_train = str(self.X_indices_arr_train[interval][-1])[0:4]
            # Adjusting for Last Batch (Unequal Sample Sizes)
            if self.X_arr_train[interval].shape[0] + self.X_arr_valid[interval].shape[0] != self.X_arr_test[interval].shape[0]:
                num_samples_test = self.X_arr_test[interval].shape[0]
                self.X_arr_train[interval] = self.X_arr_train[interval][:int(round(global_params["train_validation_split"]*num_samples_test,0))]
                self.X_arr_valid[interval] = self.X_arr_valid[interval][:int(round((1-global_params["train_validation_split"])*num_samples_test,0))]
                self.y_arr_train[interval] = self.y_arr_train[interval][:int(round(global_params["train_validation_split"]*num_samples_test,0))]
                self.y_arr_valid[interval] = self.y_arr_valid[interval][:int(round((1-global_params["train_validation_split"])*num_samples_test,0))]
            # Weight Initialization
            weights_interval = np.zeros(shape=(self.y_arr_test[interval].shape[0], self.y_arr_test[interval].shape[1]))
            # Model Training
            model = self.train(
                self.X_arr_train[interval].reshape(self.X_arr_train[interval].shape[0]*self.X_arr_train[interval].shape[1], self.X_arr_train[interval].shape[2]),
                self.y_arr_train[interval].reshape(self.y_arr_train[interval].shape[0]*self.y_arr_train[interval].shape[1], 1), 
                self.X_arr_valid[interval].reshape(self.X_arr_valid[interval].shape[0]*self.X_arr_valid[interval].shape[1], self.X_arr_valid[interval].shape[2]), 
                self.y_arr_valid[interval].reshape(self.y_arr_valid[interval].shape[0]*self.y_arr_valid[interval].shape[1], 1),
                start_year_train,
                end_year_train,
                preload_model)
            # Model Testing
            pred = model.predict(
                self.X_arr_test[interval].reshape(self.X_arr_test[interval].shape[0]*self.X_arr_test[interval].shape[1], self.X_arr_test[interval].shape[2]))
            pred = pred.reshape(self.y_arr_test[interval].shape[0], self.y_arr_test[interval].shape[1])
            # Extract Top and Bottom Assets
            top_n_assets = np.argsort(pred, axis=1)[:, -int(n/2):]
            bottom_n_assets = np.argsort(pred, axis=1)[:, :int(n/2)]
            # Update Weight Predictions
            for q, top_n_assets_i in enumerate(top_n_assets):
                weights_interval[q][top_n_assets_i] = 1
            for q, bottom_n_assets_i in enumerate(bottom_n_assets):
                weights_interval[q][bottom_n_assets_i] = -1
            weights = np.vstack([weights, weights_interval]) if len(weights) > 0 else weights_interval
        np.savetxt(f"data/predictions/csmom_mlp_weights.csv", weights, delimiter=",")
        return weights