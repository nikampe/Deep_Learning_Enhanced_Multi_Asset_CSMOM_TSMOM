import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

# Hyperparameters
from settings.global_params import global_params
from settings.hyper_params import hyperparams_fixed_csmom, hyperparams_grid_csmom
from settings.strategy_params import strategy_params_csmom

# Loss Function (ListNet)
@tf.keras.utils.register_keras_serializable()
def listnet_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, strategy_params_csmom["num_total_assets"]))
    y_pred = tf.reshape(y_pred, (-1, strategy_params_csmom["num_total_assets"]))
    y_true_probs = tf.nn.softmax(y_true, axis=-1)
    y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
    loss = -tf.reduce_sum(y_true_probs * tf.math.log(y_pred_probs + 1e-8), axis=-1)
    return tf.reduce_mean(loss)
    
# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = hyperparams_fixed_csmom["ListNet"]["early_stopping"],
    verbose = 1,
    restore_best_weights = True)

# ListNet Model
class ListNetModel(tf.keras.Model):
    def __init__(self, width_1, width_2, activation, dropout_rate):
        super(ListNetModel, self).__init__()
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

class ListNet:
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
            num_assets = strategy_params_csmom["num_total_assets"]
            # Model Building
            class CSMOMPrerankerHyperModel(kt.HyperModel):
                def build(self, hp):
                    preranker_csmom = ListNetModel(
                        width_1 = hp.Choice('width_1', values = hyperparams_grid_csmom["ListNet"]["width"]),
                        width_2 = hp.Choice('width_2', values = hyperparams_grid_csmom["ListNet"]["width"]),
                        activation = hp.Choice('activation', values = hyperparams_grid_csmom["ListNet"]["activation"]),
                        dropout_rate = hp.Choice('dropout_rate', values = hyperparams_grid_csmom["ListNet"]["dropout_rate"]))
                    preranker_csmom.compile(
                        optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values = hyperparams_grid_csmom["ListNet"]["learning_rate"])),
                        loss = listnet_loss)
                    return preranker_csmom
                def fit(self, hp, model, *args, **kwargs):
                    return model.fit(
                        *args,
                        batch_size = hp.Choice('batch_size', values = hyperparams_grid_csmom["ListNet"]["batch_size"]),
                        **kwargs,)
            # Hyperparameter Optimization
            model_tuner = kt.RandomSearch(
                CSMOMPrerankerHyperModel(),
                objective = 'val_loss',
                max_trials = hyperparams_fixed_csmom["ListNet"]["random_search_max_trials"],
                directory = 'models/hyperparameter_optimization_models',
                project_name = f'csmom_ln_{start_year_train}_{end_year_train}')
            # Hyperparameter Optimization - Grid Search
            model_tuner.search(
                X_train,
                y_train,
                epochs = hyperparams_fixed_csmom["ListNet"]["epochs"],
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
                epochs = hyperparams_fixed_csmom["ListNet"]["epochs"],
                validation_data = (X_valid, y_valid),
                callbacks=[early_stopping])
            # Save Model
            model.save(f'models/pretrained_models/csmom_ln_{start_year_train}_{end_year_train}.tf')
        else:
            model = tf.keras.models.load_model(f'models/pretrained_models/csmom_ln_{start_year_train}_{end_year_train}.tf')
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
            # Ground-Truth Decile Labels
            deciles_train = np.zeros_like(self.y_arr_train[interval])
            for x in range(self.y_arr_train[interval].shape[0]):
                daily_returns = self.y_arr_train[interval][x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_train[x, :, 0] = daily_deciles
            deciles_valid = np.zeros_like(self.y_arr_valid[interval])
            for x in range(self.y_arr_valid[interval].shape[0]):
                daily_returns = self.y_arr_valid[interval][x, :, 0]
                decile_thresholds = [np.percentile(daily_returns, t * 10) for t in range(1, 10)]
                daily_deciles = pd.cut(daily_returns, bins=[-np.inf] + decile_thresholds + [np.inf], labels=False, duplicates='drop') + 1
                deciles_valid[x, :, 0] = daily_deciles
            # Model Training
            model = self.train(
                self.X_arr_train[interval].reshape(self.X_arr_train[interval].shape[0]*self.X_arr_train[interval].shape[1], self.X_arr_train[interval].shape[2]),
                deciles_train.reshape(deciles_train.shape[0]*deciles_train.shape[1], 1), 
                self.X_arr_valid[interval].reshape(self.X_arr_valid[interval].shape[0]*self.X_arr_valid[interval].shape[1], self.X_arr_valid[interval].shape[2]), 
                deciles_valid.reshape(deciles_valid.shape[0]*deciles_valid.shape[1], 1),
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
        np.savetxt(f"data/predictions/csmom_ln_weights.csv", weights, delimiter=",")
        return weights
