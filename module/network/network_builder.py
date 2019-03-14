"""Network builder"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from module.network.bp_mll import bp_mll_loss
from module.config import Config

LAYERS = {
    'Dense':    lambda args: Dense(**args),
    'Dropout':  lambda args: Dropout(**args),
}

LOSSES = {
    "bp_mll_loss": bp_mll_loss
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
        l = LAYERS[layer['layer']](layer['args'])
        model.add(l)
    loss = config['compile']['loss']
    config['compile']['loss'] = LOSSES.get(loss, loss)
    model.compile(**config['compile'])
    return model
