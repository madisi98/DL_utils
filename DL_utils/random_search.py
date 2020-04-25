import random


def get_random_config(params):
    random_config = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            random_config[key] = value[0](*value[1])
        else:
            random_config[key] = value()
    return random_config
