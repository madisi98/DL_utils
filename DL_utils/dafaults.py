model_defaults = {
        'input_shape': None,
        'regularizer': None,
        'dropout': 0,
        'batch_norm': False,
        'activation': 'relu',
        'loss': None,
        'optimizer': 'adam',
        'metrics': None,
        'output_shape': None,
        'mode': None,
        'save_dir': '',
        'model_name': 'tmp',
        'scaler': None}

train_defaults = {
        'max_epochs': 100,
        'max_idle': 10,
        'batch_size': 1024,
        'verbose': False,
        'epochs_per_step': 5,
        'class_weights': None,
        'es_delta': 10e-4,
        'es_target': None,
}