"""Network builder"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from module.config import Config
from module.network.bp_mll import bp_mll_loss
from module.network.hamming_distance import hamming_distance

LAYERS = {
    'Dense':    lambda args: Dense(**args),
    'Dropout':  lambda args: Dropout(**args),
}

LOSSES = {
    "bp_mll_loss": bp_mll_loss
}

METRICS = {
    "hamming_distance": hamming_distance
}

def builder(x_dim, y_dim):
    """Network parser builder

    Genarates a Keras ANN architecture using a config file

    Parameters
    ----------
        config: dictionary containing the specified configuration

    Returns
    -------
        model: Keras ANN

    """
    config = Config.get()
    model = Sequential()
    config['layers'][0]['args']['input_dim'] = x_dim
    config['layers'][-1]['args']['units'] = y_dim
    for layer in config['layers']:
        lyr = LAYERS[layer['layer']](layer['args'])
        model.add(lyr)

    loss = config['compile']['loss']
    config['compile']['loss'] = LOSSES.get(loss, loss)

    metrics = []
    for metric in config['compile']['metrics']:
        metrics.append(METRICS.get(metric, metric))
    config['compile']['metrics'] = metrics

    model.compile(**config['compile'])
    return model
