# Hyperparameters Fixed
hyperparams_fixed_csmom = {
    "MLP": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "ListNet": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "LambdaMART": {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg",
        "random_search_trials": 200},
    "Transformer_Ranker": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "Transformer_Reranker": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200}}

hyperparams_fixed_tsmom = {
    "MLP": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "RNN": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "LSTM": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "Encoder_Decoder_Transformer": {
        "epochs": 100, 
        "early_stopping": 25,
        "random_search_max_trials": 200},
    "Decoder_Transformer": {
        "epochs": 100,
        "early_stopping": 25,
        "random_search_max_trials": 200}}

# Hyperpamater Grids
hyperparams_grid_csmom = {
    "MLP": {
        "mini_batch_size": [1, 2, 4, 8, 16],
        "width": [16, 32, 64, 128, 256, 512, 1024],
        "activation": ["relu", "sigmoid", "linear", "tanh"],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001]},
    "ListNet": {
        "mini_batch_size": [1, 2, 4, 8, 16],
        "width": [16, 32, 64, 128, 256, 512, 1024],
        "activation": ["relu"], # ["relu", "sigmoid", "linear", "tanh"]
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001]},
    "LambdaMART": {
        'learning_rate': [0.01, 0.001, 0.0001, 0.00001],
        'max_depth': [2, 4, 6, 8, 10]},
    "Transformer_Ranker": {
        "batch_size": [4, 8, 16, 32], # 0.13+0.37: [1, 2, 4, 8, 16]; 0.11+0.15+0.39: [2, 4, 6, 8, 10]; 0.63: [16, 32, 64, 128, 256, 512]
        "d_model": [8, 16, 32, 64, 128, 256], # 0.13+0.37: [8, 16, 32, 64, 128, 256, 512]; 0.11+0.15+0.39: [8, 16, 32, 64, 128]; 0.63: [16, 32, 64, 128, 256]
        "num_layers": [1, 2, 3], # 0.13+0.37: [1, 2, 3] 0.11+0.15+0.39: [1, 2, 3]; 0.63: [1, 2, 3]
        "num_heads": [1, 2, 4], # 0.13+0.37: [1]; 0.11+0.15+0.39: [1, 2, 4]; 0.63: [1, 2, 4]
        "dff": [8, 16, 32, 64, 128, 256], # 0.13+0.37: [8, 16, 32, 64, 128, 256, 512]; 0.11+0.15+0.39: [8, 16, 32, 64, 128]; 0.63: [32, 64, 128, 256, 512]
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5], # 0.13+0.37: [0.0, 0.2, 0.4, 0.6, 0.8]; 0.11+0.15+0.39: [0.0, 0.2, 0.4, 0.6, 0.8]; 0.63: [0.0, 0.1, 0.2, 0.3, 0.4]
        "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]},
    "Transformer_Reranker": {
        "width": [16, 32, 64, 128, 256, 512, 1024],
        "activation": ["relu"], # ["relu", "sigmoid", "linear", "tanh"],
        "batch_size": [4, 8, 16, 32],
        "d_model": [8, 16, 32, 64, 128, 256],
        "num_layers": [1, 2, 3],
        "num_heads": [1, 2 ,4],
        "dff": [8, 16, 32, 64, 128, 256],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.1, 0.01, 0.001, 0.0001, 0.00001]}}

hyperparams_grid_tsmom = {
    "MLP": {
        "mini_batch_size": [1, 2, 4, 8, 16],
        "width": [16, 32, 64, 128, 256, 512, 1024],
        "activation": ["relu", "sigmoid", "linear", "tanh"],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001]},
    "RNN": {
        "units": [8, 16, 32, 64, 128, 256],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "LSTM": {
        "units": [8, 16, 32, 64, 128, 256],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5]},
    "Encoder_Decoder_Transformer": {
        "batch_size": [64, 128, 256, 512, 1024],
        "d_model": [8, 16, 32, 64, 128, 256],
        "num_layers": [1, 2, 3], 
        "num_heads": [1, 2, 4],
        "dff": [8, 16, 32, 64, 128],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001]},
    "Decoder_Transformer": {
        "batch_size": [8, 16, 32, 64, 128],
        "d_model": [8, 16, 32, 64, 128, 256],
        "num_layers": [1, 2, 3], 
        "num_heads": [1, 2, 4],
        "dff": [8, 16, 32, 64, 128],
        "dropout_rate": [0.1, 0.2, 0.3, 0.4, 0.5],
        "learning_rate": [0.01, 0.001, 0.0001, 0.00001]}}